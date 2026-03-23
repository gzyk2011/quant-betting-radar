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

def run_global_radar():
    print("🌍 启动全球全盘口扫描雷达...")
    send_tg("🌎 *全球雷达启动*：正在扫描欧洲、美洲及低级别联赛全盘口...")

    # 极大化联赛清单 (基于 Football-Data 官方支持)
    # 注意：部分联赛如巴西甲赛季年份可能不同，这里统一用 2025-2026 或 2026
    leagues = [
        # --- 欧洲顶级 ---
        ("ENG Premier League", "2025-2026"), ("DEU Bundesliga 1", "2025-2026"),
        ("ESP La Liga", "2025-2026"), ("ITA Serie A", "2025-2026"), ("FRA Ligue 1", "2025-2026"),
        # --- 欧洲次级/活跃 ---
        ("ENG League 1", "2025-2026"), ("ENG League 2", "2025-2026"), ("ENG Championship", "2025-2026"),
        ("NLD Eredivisie", "2025-2026"), ("BEL First Division A", "2025-2026"), ("PRT Liga 1", "2025-2026"),
        ("TUR Super Lig", "2025-2026"), ("SCO Premier League", "2025-2026"),
        # --- 美洲赛区 (国际比赛日通常照常) ---
        ("BRA Serie A", "2026"), ("USA MLS", "2026"), ("MEX Liga MX", "2025-2026")
    ]

    found_any = False
    report = ["📡 *全球量化扫描结果* 📡\n"]

    for label, season in leagues:
        try:
            print(f"正在拉取: {label}...")
            df = pb.scrapers.FootballData(label, season).get_fixtures()
            df_train = df.dropna(subset=['fthg', 'ftag'])
            df_upcoming = df[df['fthg'].isna()]
            if df_upcoming.empty: continue

            # 建模
            model = pb.models.DixonColesGoalModel(df_train["goals_home"], df_train["goals_away"], df_train["team_home"], df_train["team_away"])
            model.fit(use_gradient=True, minimizer_options={"disp": False})

            for _, match in df_upcoming.iterrows():
                h, a = match['team_home'], match['team_away']
                o_h, o_over = match.get('b365_h'), match.get('b365_over_2_5')
                
                pred = model.predict(h, a)
                
                # 1. 亚盘主胜分析 (-0.5)
                if pd.notna(o_h) and 1.4 <= o_h <= 4.0:
                    edge = pred.home_win - (1/o_h)
                    if edge > 0.05:
                        found_any = True
                        report.append(f"🏆 *{label}* | 亚盘\n⚽ {h} vs {a}\n✅ 推荐: *主队-0.5* | 赔率: `{o_h}` | Edge: `{edge:.2%}`\n")

                # 2. 大小球分析 (Over 2.5)
                prob_over = 1 - (pred.total_goals(0) + pred.total_goals(1) + pred.total_goals(2))
                if pd.notna(o_over) and 1.5 <= o_over <= 3.0:
                    edge_over = prob_over - (1/o_over)
                    if edge_over > 0.05:
                        found_any = True
                        report.append(f"🏟️ *{label}* | 大小球\n⚽ {h} vs {a}\n📈 推荐: *大2.5* | 赔率: `{o_over}` | Edge: `{edge_over:.2%}`\n")
        except: continue

    if found_any:
        send_tg("\n".join(report))
    else:
        send_tg("✅ *全球扫描完毕*：当前各联赛盘口极其稳定，未发现高价值 Edge，建议继续观望。")

if __name__ == "__main__":
    run_global_radar()
