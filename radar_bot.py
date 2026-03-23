import penaltyblog as pb
import pandas as pd
import requests
import os
import warnings
warnings.filterwarnings("ignore")

# 从环境变量获取 Telegram 配置 (GitHub Actions 会注入这些变量)
TG_TOKEN = os.environ.get("AAGdETj2rEWhboXXfL5cCH-Ky6f2Y4J6NVE")
TG_CHAT_ID = os.environ.get("5097285321")

def send_telegram_message(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("未配置 TG_TOKEN，仅在控制台打印：\n", text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def run_radar():
    print("🚀 启动全网联赛价值扫描雷达...")
    messages = ["🚨 *量化足球雷达扫描报告* 🚨\n"]
    
    # 我们可以配置多个有正向 ROI 的联赛，这里以英冠为例
    leagues = [("ENG Championship", "2023-2024")] # 实际使用时改为当前赛季 "2024-2025"
    
    for comp, season in leagues:
        print(f"正在拉取 {comp} 数据...")
        df = pb.scrapers.FootballData(comp, season).get_fixtures()
        
        # 划分历史数据(用于训练)和未来数据(未踢的比赛，用于预测)
        # football-data.co.uk 中未踢的比赛 fthg (主队进球) 会是 NaN
        df_train = df.dropna(subset=['fthg', 'ftag'])
        df_upcoming = df[df['fthg'].isna()]
        
        if df_upcoming.empty:
            continue
            
        # 训练基座模型
        weights = pb.models.dixon_coles_weights(df_train["date"], xi=0.0015)
        model = pb.models.DixonColesGoalModel(
            df_train["goals_home"].values, df_train["goals_away"].values, 
            df_train["team_home"].values, df_train["team_away"].values, weights=weights
        )
        model.fit(use_gradient=True, minimizer_options={"disp": False})
        
        # 扫描未开赛的场次
        for _, match in df_upcoming.iterrows():
            home, away = match['team_home'], match['team_away']
            odds_h, odds_d, odds_a = match['b365_h'], match['b365_d'], match['b365_a']
            
            # 如果没有提前开出赔率，跳过
            if pd.isna(odds_h) or odds_h <= 1.0:
                continue
                
            try:
                pred = model.predict(home, away)
            except ValueError:
                continue

            outcomes = [("主胜", odds_h, pred.home_win), ("平局", odds_d, pred.draw), ("客胜", odds_a, pred.away_win)]
            
            for name, odds, prob in outcomes:
                # 策略 1：黄金赔率区间 1.5 ~ 3.5
                if odds < 1.5 or odds > 3.5:
                    continue
                    
                implied_prob = 1 / odds
                edge = prob - implied_prob
                
                # 策略 2：Edge > 5%
                if edge > 0.05:
                    # 策略 3：1/20 凯利风控
                    kc = pb.betting.kelly_criterion(odds, prob, 0.05).stake
                    if kc > 0:
                        ev = (prob * (odds - 1)) - (1 - prob)
                        msg = (
                            f"⚽ *{home} vs {away}*\n"
                            f"💡 建议买入: *{name}*\n"
                            f"🏦 盘口赔率: `{odds}`\n"
                            f"📊 模型胜率: `{prob:.2%}` (市场: `{implied_prob:.2%}`)\n"
                            f"🔥 Edge优势: `{edge:.2%}` | EV: `{ev:.2%}`\n"
                            f"💰 推荐仓位: `{float(kc):.2%}`\n"
                            "------------------------"
                        )
                        messages.append(msg)
                        print(f"发现目标：{home} vs {away} - {name}")
                        
    if len(messages) > 1:
        send_telegram_message("\n".join(messages))
    else:
        print("当前无符合策略的价值投注。")
        # send_telegram_message("本期扫描完毕，未发现高价值盘口，继续管住手！")

if __name__ == "__main__":
    run_radar()