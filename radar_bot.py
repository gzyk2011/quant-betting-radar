import penaltyblog as pb
import pandas as pd
import requests
import os
import warnings
from io import StringIO

warnings.filterwarnings("ignore")

# --- 核心配置 ---
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")
MIN_EDGE = 0.07  # 7% 利润门槛

def send_tg(text):
    if not TG_TOKEN or not TG_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"})

def get_realtime_data():
    """抓取全维度赔率表：包含欧指、大小球、亚盘"""
    url = "https://www.football-data.co.uk/fixtures.csv"
    try:
        r = requests.get(url)
        df = pd.read_csv(StringIO(r.text))
        # 字段映射
        df = df.rename(columns={
            'HomeTeam': 'team_home', 'AwayTeam': 'team_away',
            'BbAvH': 'o_h', 'BbAvD': 'o_d', 'BbAvA': 'o_a',       # 欧指平均
            'BbAv>2.5': 'o_over', 'BbAv<2.5': 'o_under',          # 大小球平均
            'BbAHh': 'ah_line', 'BbAvAHH': 'o_ah_h', 'BbAvAHA': 'o_ah_a' # 亚盘线及赔率
        })
        return df
    except: return pd.DataFrame()

def run_ultimate_sniper():
    print("🛰️ 启动[三位一体]量化狙击雷达...")
    send_tg("🎯 *量化狙击手 V6.0*：正在扫描 [主胜+大小球+亚盘] 复合信号...")

    df_live = get_realtime_data()
    if df_live.empty: return

    leagues_config = [
        {"div": "E0", "label": "ENG Premier League", "xi": 0.005},
        {"div": "E1", "label": "ENG League 1", "xi": 0.001},
        {"div": "E2", "label": "ENG League 2", "xi": 0.001},
        {"div": "D2", "label": "DEU Bundesliga 2", "xi": 0.001},
        {"div": "SC2", "label": "SCO Division 2", "xi": 0.001}
    ]

    found_any = False
    report = ["🎰 *全维度价值预警* 🎰\n"]

    for cfg in leagues_config:
        try:
            # 1. 训练模型
            df_hist = pb.scrapers.FootballData(cfg["label"], "2025-2026").get_fixtures()
            df_train = df_hist.dropna(subset=['fthg', 'ftag'])
            weights = pb.models.dixon_coles_weights(df_train["date"], xi=cfg["xi"])
            model = pb.models.DixonColesGoalModel(df_train["goals_home"], df_train["goals_away"], df_train["team_home"], df_train["team_away"], weights=weights)
            model.fit(use_gradient=True)

            # 2. 匹配实时数据
            current_odds = df_live[df_live['Div'] == cfg['div']]
            
            for _, m in current_odds.iterrows():
                h, a = m['team_home'], m['team_away']
                try:
                    pred = model.predict(h, a)
                    
                    # --- [1] 主胜分析 ---
                    if pd.notna(m['o_h']):
                        implied_h = pb.implied.calculate_implied([m['o_h'], m['o_d'], m['o_a']], method="shin").probabilities[0]
                        edge_h = pred.home_win - implied_h
                        if edge_h > MIN_EDGE:
                            found_any = True
                            report.append(f"🚩 *{cfg['label']}* | 主胜\n⚽ {h} vs {a}\n📈 赔率: `{m['o_h']:.2f}` | 优势: `{edge_h:.2%}`\n")

                    # --- [2] 大小球分析 ---
                    if pd.notna(m['o_over']):
                        implied_over = pb.implied.calculate_implied([m['o_over'], m['o_under']], method="shin").probabilities[0]
                        prob_over = 1 - (pred.total_goals(0) + pred.total_goals(1) + pred.total_goals(2))
                        edge_over = prob_over - implied_over
                        if edge_over > MIN_EDGE:
                            found_any = True
                            report.append(f"🏟️ *{cfg['label']}* | 大2.5\n⚽ {h} vs {a}\n📈 赔率: `{m['o_over']:.2f}` | 优势: `{edge_over:.2%}`\n")

                    # --- [3] 亚盘分析 (核心逻辑) ---
                    if pd.notna(m['o_ah_h']) and pd.notna(m['ah_line']):
                        # 计算模型在指定让球线下的胜率
                        line = m['ah_line']
                        prob_ah_win = 0
                        prob_ah_push = 0
                        for h_g in range(10):
                            for a_g in range(10):
                                p = pred.home_goal_probs[h_g] * pred.away_goal_probs[a_g]
                                res = h_g - a_g + line
                                if res > 0: prob_ah_win += p
                                elif res == 0: prob_ah_push += p
                        
                        # 剔除走盘概率后的模型净胜率
                        adj_model_ah_prob = prob_ah_win / (1 - prob_ah_push) if prob_ah_push < 1 else 0
                        implied_ah = pb.implied.calculate_implied([m['o_ah_h'], m['o_ah_a']], method="shin").probabilities[0]
                        edge_ah = adj_model_ah_prob - implied_ah
                        
                        if edge_ah > MIN_EDGE:
                            found_any = True
                            report.append(f"🛡️ *{cfg['label']}* | 亚盘 {line}\n⚽ {h} vs {a}\n📈 赔率: `{m['o_ah_h']:.2f}` | 优势: `{edge_ah:.2%}`\n")
                except: continue
        except: continue

    if found_any:
        send_tg("\n".join(report))
    else:
        send_tg("✅ *扫描完成*：当前暂无符合 7% 门槛的复合价值信号。")

if __name__ == "__main__":
    run_ultimate_sniper()
