import penaltyblog as pb
import pandas as pd
import requests
import os
import warnings
from io import StringIO

warnings.filterwarnings("ignore")

# --- 核心策略配置 ---
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")
MIN_EDGE = 0.07      # 7% 利润门槛
KELLY_FRAC = 0.05    # 凯利缩放系数（建议 0.05-0.1，越低越稳）

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
        df = df.rename(columns={
            'HomeTeam': 'team_home', 'AwayTeam': 'team_away',
            'BbAvH': 'o_h', 'BbAvD': 'o_d', 'BbAvA': 'o_a',
            'BbAv>2.5': 'o_over', 'BbAv<2.5': 'o_under',
            'BbAHh': 'ah_line', 'BbAvAHH': 'o_ah_h', 'BbAvAHA': 'o_ah_a'
        })
        return df
    except: return pd.DataFrame()

def run_ultimate_sniper():
    print("🛰️ 启动[凯利仓位版]量化狙击雷达...")
    send_tg("🎯 *量化狙击手 V6.1*：实时扫描已开启，包含[凯利准则]仓位建议。")

    df_live = get_realtime_data()
    if df_live.empty: return

    # 黄金联赛清单 (基于回测 ROI 表现)
    leagues_config = [
        {"div": "E0", "label": "ENG Premier League", "xi": 0.005},
        {"div": "E1", "label": "ENG League 1", "xi": 0.001},
        {"div": "E2", "label": "ENG League 2", "xi": 0.001},
        {"div": "D2", "label": "DEU Bundesliga 2", "xi": 0.001},
        {"div": "SC2", "label": "SCO Division 2", "xi": 0.001}
    ]

    found_any = False
    report = ["🎰 *量化价值实时预警 (带仓位建议)* 🎰\n"]

    for cfg in leagues_config:
        try:
            # 1. 建模
            df_hist = pb.scrapers.FootballData(cfg["label"], "2025-2026").get_fixtures()
            df_train = df_hist.dropna(subset=['fthg', 'ftag'])
            weights = pb.models.dixon_coles_weights(df_train["date"], xi=cfg["xi"])
            model = pb.models.DixonColesGoalModel(df_train["goals_home"], df_train["goals_away"], df_train["team_home"], df_train["team_away"], weights=weights)
            model.fit(use_gradient=True)

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
                            # 凯利计算
                            kelly = pb.betting.kelly_criterion(m['o_h'], pred.home_win, KELLY_FRAC)
                            report.append(f"🚩 *{cfg['label']}* | 主胜\n⚽ {h} vs {a}\n📈 赔率: `{m['o_h']:.2f}` | 优势: `{edge_h:.2%}`\n💰 *建议仓位: {kelly.stake:.2%}*")

                    # --- [2] 大小球分析 ---
                    if pd.notna(m['o_over']):
                        implied_over = pb.implied.calculate_implied([m['o_over'], m['o_under']], method="shin").probabilities[0]
                        prob_over = 1 - (pred.total_goals(0) + pred.total_goals(1) + pred.total_goals(2))
                        edge_over = prob_over - implied_over
                        if edge_over > MIN_EDGE:
                            found_any = True
                            kelly_ov = pb.betting.kelly_criterion(m['o_over'], prob_over, KELLY_FRAC)
                            report.append(f"🏟️ *{cfg['label']}* | 大2.5\n⚽ {h} vs {a}\n📈 赔率: `{m['o_over']:.2f}` | 优势: `{edge_over:.2%}`\n💰 *建议仓位: {kelly_ov.stake:.2%}*")

                    # --- [3] 亚盘分析 ---
                    if pd.notna(m['o_ah_h']) and pd.notna(m['ah_line']):
                        line = m['ah_line']
                        p_win, p_push = 0, 0
                        for h_g in range(10):
                            for a_g in range(10):
                                p = pred.home_goal_probs[h_g] * pred.away_goal_probs[a_g]
                                res = h_g - a_g + line
                                if res > 0: p_win += p
                                elif res == 0: p_push += p
                        
                        adj_prob = p_win / (1 - p_push) if p_push < 1 else 0
                        implied_ah = pb.implied.calculate_implied([m['o_ah_h'], m['o_ah_a']], method="shin").probabilities[0]
                        edge_ah = adj_prob - implied_ah
                        
                        if edge_ah > MIN_EDGE:
                            found_any = True
                            kelly_ah = pb.betting.kelly_criterion(m['o_ah_h'], adj_prob, KELLY_FRAC)
                            report.append(f"🛡️ *{cfg['label']}* | 亚盘 {line}\n⚽ {h} vs {a}\n📈 赔率: `{m['o_ah_h']:.2f}` | 优势: `{edge_ah:.2%}`\n💰 *建议仓位: {kelly_ah.stake:.2%}*")
                    
                    if found_any: report.append("---")
                except: continue
        except: continue

    if found_any:
        send_tg("\n".join(report))
    else:
        print("✅ 扫描完成：暂无符合优势的信号。")

if __name__ == "__main__":
    run_ultimate_sniper()
