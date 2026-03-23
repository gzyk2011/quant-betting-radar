import penaltyblog as pb
import pandas as pd
import requests
import os
import warnings

warnings.filterwarnings("ignore")

TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

def send_tg(text):
    if not TG_TOKEN or not TG_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"})

def run_ultimate_radar():
    print("📡 启动全功能矩阵雷达：全盘口扫描中...")
    send_tg("💹 *全盘口量化雷达*：已开启亚盘/大小球深度计算...")

    leagues = [
        ("ENG League 1", "2025-2026"), ("ENG League 2", "2025-2026"),
        ("ENG Premier League", "2025-2026"), ("DEU Bundesliga 1", "2025-2026"),
        ("ESP La Liga", "2025-2026"), ("ITA Serie A", "2025-2026")
    ]

    found_any = False
    report = ["🎰 *全盘口价值预警* 🎰\n"]

    for label, season in leagues:
        try:
            df = pb.scrapers.FootballData(label, season).get_fixtures()
            df_train = df.dropna(subset=['fthg', 'ftag'])
            df_upcoming = df[df['fthg'].isna()]
            if df_upcoming.empty: continue

            # 1. 训练 Dixon-Coles 获取进球分布能力
            model = pb.models.DixonColesGoalModel(df_train["goals_home"], df_train["goals_away"], df_train["team_home"], df_train["team_away"])
            model.fit(use_gradient=True)

            for _, match in df_upcoming.iterrows():
                h, a = match['team_home'], match['team_away']
                # 获取不同玩法的赔率
                odds = {
                    "主胜": match.get('b365_h'),
                    "大2.5": match.get('b365_over_2_5'),
                    "小2.5": match.get('b365_under_2_5')
                }
                
                # 获取预测概率分布 (0-10球的矩阵)
                pred = model.predict(h, a)
                
                # --- A. 亚盘核心转换 (利用本地源码中的概率累加逻辑) ---
                # 计算主队让步胜率
                ah_0 = pred.home_win / (pred.home_win + pred.away_win) # 平手盘(0)去掉平局后的胜率
                ah_minus_0_5 = pred.home_win # 让半球(-0.5)等于直接胜
                ah_minus_1_5 = sum(pred.home_goal_probs[i] * sum(pred.away_goal_probs[j] for j in range(i-1)) for i in range(2, 10)) # 让球1.5
                
                # --- B. 大小球全盘口扫描 ---
                # 计算各种进球数的概率
                prob_over_1_5 = 1 - (pred.total_goals(0) + pred.total_goals(1))
                prob_over_2_5 = 1 - (pred.total_goals(0) + pred.total_goals(1) + pred.total_goals(2))
                prob_over_3_5 = 1 - (pred.total_goals(0) + pred.total_goals(1) + pred.total_goals(2) + pred.total_goals(3))

                # --- C. 价值识别 (示例：大2.5球价值) ---
                if pd.notna(odds["大2.5"]):
                    edge = prob_over_2_5 - (1/odds["大2.5"])
                    if edge > 0.05:
                        found_any = True
                        report.append(f"🚩 *大小球* | {label}\n⚽ {h} vs {a}\n📈 推荐: *大 2.5* | 赔率: `{odds['大2.5']}` | Edge: `{edge:.2%}`\n")

                # --- D. 亚盘价值识别 (主胜 Edge) ---
                if pd.notna(odds["主胜"]):
                    edge_h = ah_minus_0_5 - (1/odds["主胜"])
                    if edge_h > 0.05:
                        found_any = True
                        report.append(f"🚩 *亚盘(换算)* | {label}\n⚽ {h} vs {a}\n✅ 推荐: *主队 -0.5* | 赔率: `{odds['主胜']}` | Edge: `{edge_h:.2%}`\n")

        except Exception as e:
            print(f"Error in {label}: {e}")
            continue

    if found_any:
        send_tg("\n".join(report))
    else:
        send_tg("✅ *全盘口扫描完毕*：当前英甲/英乙等活跃赛场暂无符合 Edge > 5% 的亚盘或大小球机会。")

if __name__ == "__main__":
    run_ultimate_radar()
