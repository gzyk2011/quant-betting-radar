import penaltyblog as pb
import pandas as pd
import requests
import os
import warnings

warnings.filterwarnings("ignore")

# 从 GitHub Secrets 获取配置
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

def send_tg(text):
    if not TG_TOKEN: return print(text)
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"})

def run_ultimate_radar():
    print("🚀 正在启动全功能量化雷达...")
    messages = ["🌟 *PenaltyBlog 终极策略报告* 🌟\n"]
    
    # 定义要扫描的所有顶级联赛
    leagues = [
        ("ENG Premier League", "E0"), ("ENG Championship", "E1"),
        ("GER Bundesliga", "D1"), ("ITA Serie A", "I1"),
        ("ESP La Liga", "SP1"), ("FRA Ligue 1", "F1")
    ]
    
    for label, code in leagues:
        try:
            print(f"分析中: {label}...")
            # 1. 抓取数据
            df = pb.scrapers.FootballData(label, "2023-2024").get_fixtures()
            df_train = df.dropna(subset=['fthg', 'ftag'])
            df_upcoming = df[df['fthg'].isna()]
            
            if df_upcoming.empty: continue

            # 2. 整合 Elo 评级 (长期实力维度)
            elo = pb.ratings.Elo()
            elo.update(df_train["goals_home"], df_train["goals_away"], df_train["team_home"], df_train["team_away"])
            current_ratings = elo.get_ratings()

            # 3. 训练 Dixon-Coles 模型 (进球率维度)
            weights = pb.models.dixon_coles_weights(df_train["date"], xi=0.0015)
            model = pb.models.DixonColesGoalModel(
                df_train["goals_home"].values, df_train["goals_away"].values, 
                df_train["team_home"].values, df_train["team_away"].values, weights=weights
            )
            model.fit(use_gradient=True)

            # 4. 扫描预测
            for _, match in df_upcoming.iterrows():
                h, a = match['team_home'], match['team_away']
                odds = [match.get('b365_h'), match.get('b365_d'), match.get('b365_a')]
                
                if any(pd.isna(o) or o < 1.5 or o > 3.5 for o in odds): continue

                # 使用 Shin 方法精准剔除抽水
                implied = pb.implied.calculate_implied(odds, method="shin")
                pred = model.predict(h, a)
                
                # 策略核心：Edge > 5% 且 Elo 评级优势
                edge = pred.home_win - implied.probabilities[0]
                elo_diff = current_ratings.get(h, 1000) - current_ratings.get(a, 1000)
                
                if edge > 0.05 and elo_diff > 50:
                    kc = pb.betting.kelly_criterion(odds[0], pred.home_win, 0.05).stake
                    messages.append(f"🏆 *{label}*\n🔥 目标: {h} vs {a}\n✅ 推荐: 主胜\n💰 赔率: `{odds[0]}` | Edge: `{edge:.2%}`\n📊 Elo优势: `{int(elo_diff)}` | 推荐仓位: `{float(kc):.2%}`\n")
                    
        except Exception as e:
            print(f"跳过 {label}: {e}")

    send_tg("\n".join(messages) if len(messages) > 1 else "✅ *雷达扫描完毕*\n当前全网无复合高ROI策略的场次，继续保持耐心。")

if __name__ == "__main__":
    run_ultimate_radar()
    if __name__ == "__main__":
    # 在运行正式逻辑前，先发一条测试消息
    send_tg("🔔 *雷达测试*：云端连接成功，正在开始扫描五大联赛...")
    
    run_radar()
    
    # 扫描结束后再发一条
    send_tg("🏁 *扫描结束*：已完成全部联赛比对。")
