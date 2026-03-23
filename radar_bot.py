import penaltyblog as pb
import pandas as pd
import requests
import os

TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

def send_tg(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"DEBUG: {text}")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    res = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"})
    print(f"TG 发送尝试，状态码: {res.status_code}")

def run_radar():
    # 修正后的联赛名称
    leagues = [
        ("ENG Premier League", "2025-2026"),
        ("ENG Championship", "2025-2026"),
        ("GER 1. Bundesliga", "2025-2026"), # 修正德甲名
        ("ITA Serie A", "2025-2026"),
        ("ESP La Liga", "2025-2026"),
        ("FRA Ligue 1", "2025-2026")
    ]
    
    found_any = False
    report = ["🚨 *3月量化扫描报告* 🚨\n"]

    for label, season in leagues:
        try:
            print(f"正在分析: {label}")
            df = pb.scrapers.FootballData(label, season).get_fixtures()
            
            # 为了测试，我们扫描最近 3 天已经踢完的比赛，看看模型准不准
            # 正常运行时，这里应该扫描 df[df['fthg'].isna()]
            df_train = df.dropna(subset=['fthg', 'ftag'])
            
            # 这里的逻辑：如果有未来比赛就扫未来，没有就提示没数据
            df_upcoming = df[df['fthg'].isna()]
            
            if df_upcoming.empty:
                print(f"{label} 目前处于国际比赛日停赛期")
                continue

            model = pb.models.DixonColesGoalModel(df_train["goals_home"], df_train["goals_away"], df_train["team_home"], df_train["team_away"])
            model.fit(use_gradient=True)

            for _, match in df_upcoming.iterrows():
                h, a, o_h = match['team_home'], match['team_away'], match.get('b365_h')
                if pd.isna(o_h): continue

                pred = model.predict(h, a)
                edge = pred.home_win - (1/o_h)
                
                if edge > 0.02:
                    found_any = True
                    report.append(f"⚽ *{label}*\n{h} vs {a}\n建议: 主胜 | 赔率: `{o_h}`\n优势: `{edge:.2%}`\n")
        except Exception as e:
            print(f"{label} 出错: {e}")

    if found_any:
        send_tg("\n".join(report))
    else:
        send_tg("✅ *扫描完成*：当前正值国际比赛日，五大联赛暂无赛程。雷达将持续监控下周末联赛回归！")

if __name__ == "__main__":
    # 强制发送一条测试，如果这条你都没收到，100% 是 Secrets 填错了
    send_tg("🤖 *雷达状态确认*：云端连接正常，开始扫描...")
    run_radar()
