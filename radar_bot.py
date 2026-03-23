import penaltyblog as pb
import pandas as pd
import requests
import os
import warnings

warnings.filterwarnings("ignore")

# --- 配置区 ---
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")
KELLY_FRACTION = 0.05  # 凯利系数：设为 0.05 意味着使用“小凯利”，更稳健

def send_tg(text):
    if not TG_TOKEN or not TG_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"})

def run_ultimate_radar():
    print("🚀 启动全功能量化雷达 [Pounder 加速版]...")
    send_tg("🕵️ *终极雷达启动*：Pounder 模型 + 凯利仓位控制已就位。")

    # 极大化联赛清单
    leagues = [
        ("ENG Premier League", "2025-2026"), ("DEU Bundesliga 1", "2025-2026"),
        ("ESP La Liga", "2025-2026"), ("ITA Serie A", "2025-2026"), ("FRA Ligue 1", "2025-2026"),
        ("ENG League 1", "2025-2026"), ("ENG League 2", "2025-2026"), ("ENG Championship", "2025-2026"),
        ("NLD Eredivisie", "2025-2026"), ("BEL First Division A", "2025-2026"), ("PRT Liga 1", "2025-2026"),
        ("TUR Super Lig", "2025-2026"), ("SCO Premier League", "2025-2026"),
        ("BRA Serie A", "2026"), ("USA MLS", "2026"), ("MEX Liga MX", "2025-2026")
    ]

    found_any = False
    report = ["📡 *量化扫描实时信号* 📡\n"]

    for label, season in leagues:
        try:
            df = pb.scrapers.FootballData(label, season).get_fixtures()
            df_train = df.dropna(subset=['fthg', 'ftag'])
            df_upcoming = df[df['fthg'].isna()]
            if df_upcoming.empty: continue

            # --- 1. 使用 Pounder 模型加速 ---
            # 它比 Dixon-Coles 快，非常适合多联赛并发扫描
            model = pb.models.PounderGoalModel(
                df_train["goals_home"], df_train["goals_away"], 
                df_train["team_home"], df_train["team_away"]
            )
            model.fit()

            for _, match in df_upcoming.iterrows():
                h, a = match['team_home'], match['team_away']
                o_h, o_over = match.get('b365_h'), match.get('b365_over_2_5')
                
                pred = model.predict(h, a)
                
                # --- 2. 亚盘/主胜分析 + 凯利公式 ---
                if pd.notna(o_h) and 1.4 <= o_h <= 4.0:
                    prob = pred.home_win
                    edge = prob - (1/o_h)
                    
                    if edge > 0.05:
                        # 计算凯利建议仓位
                        kelly = pb.betting.kelly_criterion(o_h, prob, KELLY_FRACTION)
                        if kelly.stake > 0:
                            found_any = True
                            report.append(f"🏆 *{label}* | 亚盘\n⚽ {h} vs {a}\n✅ 推荐: *主队-0.5* | 赔率: `{o_h}`\n📊 Edge: `{edge:.2%}` | 建议仓位: `{kelly.stake:.2%}`\n")

                # --- 3. 大小球分析 (Over 2.5) + 凯利公式 ---
                prob_over = 1 - (pred.total_goals(0) + pred.total_goals(1) + pred.total_goals(2))
                if pd.notna(o_over) and 1.5 <= o_over <= 3.0:
                    edge_over = prob_over - (1/o_over)
                    if edge_over > 0.05:
                        kelly_o = pb.betting.kelly_criterion(o_over, prob_over, KELLY_FRACTION)
                        if kelly_o.stake > 0:
                            found_any = True
                            report.append(f"🏟️ *{label}* | 大小球\n⚽ {h} vs {a}\n📈 推荐: *大2.5* | 赔率: `{o_over}`\n📊 Edge: `{edge_over:.2%}` | 建议仓位: `{kelly_o.stake:.2%}`\n")
                            
        except Exception as e:
            print(f"跳过 {label}: {e}")
            continue

    if found_any:
        send_tg("\n".join(report))
    else:
        send_tg("✅ *扫描完成*：当前全球各联赛暂无符合高优势+凯利准则的信号。")

if __name__ == "__main__":
    run_ultimate_radar()
