import penaltyblog as pb
import pandas as pd
import requests
import os
import warnings
from io import StringIO

warnings.filterwarnings("ignore")

TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

def send_tg(text):
    if not TG_TOKEN or not TG_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"})

def get_realtime_odds():
    """从 football-data.co.uk 直接抓取本周最新赔率表"""
    url = "https://www.football-data.co.uk/fixtures.csv"
    try:
        r = requests.get(url)
        df = pd.read_csv(StringIO(r.text))
        # 统一列名映射
        df = df.rename(columns={'HomeTeam': 'team_home', 'AwayTeam': 'team_away', 'B365H': 'o_h', 'B365D': 'o_d', 'B365A': 'o_a'})
        return df
    except:
        return pd.DataFrame()

def run_elite_radar():
    print("🛰️ 启动[全能实时版]量化雷达...")
    send_tg("🤖 *实时狙击手启动*：正在抓取本周最新赔率并对比模型...")

    # 获取本周全联赛最新赔率
    df_odds = get_realtime_odds()
    if df_odds.empty:
        send_tg("⚠️ 无法获取本周最新赔率文件。")
        return

    # 你的黄金联赛配置 (基于回测结果)
    leagues_config = [
        {"div": "E0", "label": "ENG Premier League", "xi": 0.005},
        {"div": "E1", "label": "ENG League 1", "xi": 0.001},
        {"div": "E2", "label": "ENG League 2", "xi": 0.001},
        {"div": "D2", "label": "DEU Bundesliga 2", "xi": 0.001}
    ]

    found_any = False
    report = ["🎰 *量化价值实时预警* 🎰\n"]

    for cfg in leagues_config:
        try:
            # 1. 训练模型 (获取历史数据)
            df_history = pb.scrapers.FootballData(cfg["label"], "2025-2026").get_fixtures()
            df_train = df_history.dropna(subset=['fthg', 'ftag'])
            weights = pb.models.dixon_coles_weights(df_train["date"], xi=cfg["xi"])
            model = pb.models.DixonColesGoalModel(df_train["goals_home"], df_train["goals_away"], df_train["team_home"], df_train["team_away"], weights=weights)
            model.fit(use_gradient=True)

            # 2. 从本周赔率表中筛选出该联赛的比赛
            current_league_odds = df_odds[df_odds['Div'] == cfg['div']]
            
            for _, match in current_league_odds.iterrows():
                h, a = match['team_home'], match['team_away']
                o_h, o_d, o_a = match['o_h'], match['o_d'], match['o_a']
                
                if pd.notna(o_h):
                    try:
                        pred = model.predict(h, a)
                        # Shin 剔除抽水
                        implied = pb.implied.calculate_implied([o_h, o_d, o_a], method="shin")
                        real_prob = implied.probabilities[0]
                        
                        edge = pred.home_win - real_prob
                        
                        if edge > 0.07: # 7% 利润安全垫
                            found_any = True
                            fair_odds = 1 / pred.home_win
                            report.append(f"💎 *{cfg['label']}*\n🏟️ {h} vs {a}\n📈 市场赔率: `{o_h}` | 公平赔率: `{fair_odds:.2f}`\n📊 预期利润: `{edge:.2%}`\n")
                    except: continue # 可能是队名对不上的错误，跳过
        except: continue

    if found_any:
        send_tg("\n".join(report))
    else:
        send_tg("✅ *扫描完成*：本周暂无符合 7% 优势的比赛。")

if __name__ == "__main__":
    run_elite_radar()
