import penaltyblog as pb
import pandas as pd
import requests
import os
import warnings

warnings.filterwarnings("ignore")

TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

def send_tg(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("❌ 错误：GitHub Secrets 没配置好，找不到 Token 或 ID")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("✅ Telegram 消息发送成功！")
    else:
        print(f"❌ Telegram 发送失败！错误码：{response.status_code}")
        print(f"❌ 错误详情：{response.text}") # 这行会告诉你到底是 Token 错了还是没点 Start

def run_radar():
    print("🚀 启动基于官方清单的全球扫描...")
    # 发送一个启动信号，证明云端活着
    send_tg("📡 *量化雷达已启动*：正在根据官方清单扫描英甲/英乙等活跃赛事...")
    
    # 严格按照你刚刚跑出来的 test.py 清单进行配置
    leagues = [
        ("ENG League 1", "2025-2026"),
        ("ENG League 2", "2025-2026"),
        ("ENG Premier League", "2025-2026"),
        ("DEU Bundesliga 1", "2025-2026"),
        ("ESP La Liga", "2025-2026"),
        ("ITA Serie A", "2025-2026"),
        ("FRA Ligue 1", "2025-2026"),
        ("NLD Eredivisie", "2025-2026"),
        ("BEL First Division A", "2025-2026")
    ]
    
    found_any = False
    report = ["🚨 *实时价值投注建议* 🚨\n"]

    for label, season in leagues:
        try:
            print(f"正在拉取官方联赛: {label}...")
            df = pb.scrapers.FootballData(label, season).get_fixtures()
            
            df_train = df.dropna(subset=['fthg', 'ftag'])
            df_upcoming = df[df['fthg'].isna()]
            
            if df_upcoming.empty:
                continue

            # 训练模型
            model = pb.models.DixonColesGoalModel(df_train["goals_home"], df_train["goals_away"], df_train["team_home"], df_train["team_away"])
            model.fit(use_gradient=True, minimizer_options={"disp": False})

            for _, match in df_upcoming.iterrows():
                h, a = match['team_home'], match['team_away']
                o_h = match.get('b365_h')
                
                # 设置较宽松的门槛（2%），确保能在国际比赛日期间抓到低级别联赛机会
                if pd.notna(o_h) and 1.3 <= o_h <= 4.5:
                    pred = model.predict(h, a)
                    edge = pred.home_win - (1/o_h)
                    
                    if edge > 0.02: 
                        found_any = True
                        report.append(f"🏆 *{label}*\n⚽ {h} vs {a}\n📈 建议: 主胜 | 赔率: `{o_h}`\n🔥 优势: `{edge:.2%}`\n")
        except Exception as e:
            print(f"跳过 {label}: 名称不匹配或数据源错误 ({e})")
            continue

    if found_any:
        send_tg("\n".join(report))
    else:
        send_tg("✅ *扫描完成*：当前活跃联赛（英甲/英乙等）暂无符合策略的场次。")

if __name__ == "__main__":
    run_radar()
