import statsmodels.api as sm
from statsmodels import regression
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import scipy.stats as st
from scipy.stats import norm
import time
import datetime
from datetime import date
from jqdata import *
import math
import jqdata
from jqdata import finance
from jqfactor import get_factor_values
from jqfactor import Factor, calc_factors
import talib

'''
================================================================================
总体回测前
================================================================================
'''


# 总体回测前要做的事情
def initialize(context):
    set_params(context)  # 1设置策参数
    set_variables()  # 2设置中间变量
    set_backtest()  # 3设置回测条件
    run_daily(func, time='9:30')


# 1
# 设置策参数
def set_params(context):
    g.tc = 5  # 调仓频率
    g.yb = 2  # 样本长度,过滤前g.yb天未停牌的股票

    g.days = 0
    g.stock_sell = []
    g.everyStock = context.portfolio.portfolio_value / 10


# 2
# 设置中间变量
def set_variables():
    g.t = 0  # 记录连续回测天数
    g.if_trade = False  # 当天是否交易

    today = date.today()  # 取当日时间xxxx-xx-xx
    a = get_all_trade_days()  # 取所有交易日
    g.ATD = [''] * len(a)  # 获得len(a)维的单位向量
    for i in range(0, len(a)):
        g.ATD[i] = a[i].isoformat()  # 转换所有交易日为iso格式
        # 列表会取到2016-12-30，现在需要将大于今天的列表全部砍掉
        if today <= a[i]:
            break
    g.ATD = g.ATD[:i]  # iso格式的交易日


# 3
# 设置回测条件
def set_backtest():
    set_benchmark('399905.XSHE')
    set_option('use_real_price', True)
    log.set_level('order', 'error')


'''
================================================================================
每天开盘前
================================================================================
'''


# 每天开盘前要做的事情
def before_trading_start(context):
    if g.t % g.tc == 0:
        # 每g.tc天，交易一次行
        g.if_trade = True
        # 设置手续费与手续费
        set_slip_fee(context)

        Q1 = get_index_stocks('000852.XSHG')
        Q2 = get_index_stocks('399012.XSHE')
        tem_index = Q1 + Q2
        tem_index = list(get_all_securities(['stock']).index)
        tem_index = set(tem_index)
        print(len(tem_index))
        g.all_stocks = set_feasible_stocks(tem_index, g.yb, context)
        g.all_stocks = sorted(g.all_stocks)  # 股票代码排序
    g.t += 1


# 4 根据不同的时间段设置滑点与手续费
def set_slip_fee(context):
    # 将滑点设置为0.00246
    set_slippage(PriceRelatedSlippage(0.000001))
    # 根据不同的时间段设置手续费
    dt = context.current_dt
    log.info(type(context.current_dt))
    set_commission(PerTrade(buy_cost=0.0000005, sell_cost=0.00005, min_cost=0))


# 5
# 设置可行股票池：
# 过滤掉当日停牌的股票,且筛选出前days天未停牌股票
# 输入：stock_list-list类型,样本天数days-int类型，context（见API）
# 输出：颗星股票池-list类型
def set_feasible_stocks(stock_list, days, context):
    # 得到是否停牌信息的dataframe，停牌的1，未停牌得0
    suspened_info_df = \
    get_price(list(stock_list), start_date=context.current_dt, end_date=context.current_dt, frequency='daily',
              fields='paused')['paused'].T
    # 过滤停牌股票 返回dataframe
    unsuspened_index = suspened_info_df.iloc[:, 0] < 1
    # 得到当日未停牌股票的代码list:
    unsuspened_stocks = suspened_info_df[unsuspened_index].index
    # 进一步，筛选出前days天未曾停牌的股票list:
    feasible_stocks = []
    current_data = get_current_data()
    for stock in unsuspened_stocks:
        if sum(attribute_history(stock, days, unit='1d', fields=('paused'), skip_paused=False))[0] == 0:
            feasible_stocks.append(stock)
    # 过滤涨停的股票
    last_prices = history(1, unit='1m', field='close', security_list=feasible_stocks)
    # 已存在于持仓的股票即使涨停也不过滤，避免此股票再次可买，但因被过滤而导致选择别的股票
    feasible_stocks = [stock for stock in feasible_stocks if stock in context.portfolio.positions.keys()
                       or last_prices[stock][-1] < current_data[stock].high_limit]
    # 过滤跌停的股票
    feasible_stocks = [stock for stock in feasible_stocks if stock in context.portfolio.positions.keys()
                       or last_prices[stock][-1] > current_data[stock].low_limit]

    # 过滤ST及其他具有退市标签的股票
    feasible_stocks = [stock for stock in feasible_stocks
                       if not current_data[stock].is_st
                       and 'ST' not in current_data[stock].name
                       and '*' not in current_data[stock].name
                       and '退' not in current_data[stock].name]
    # 上市时间大于3年，考虑财务数据
    feasible_stocks = [stock for stock in feasible_stocks if
                       ((context.current_dt.date() - get_security_info(stock).start_date).days) > 365]
    return feasible_stocks


'''
================================================================================
每天交易时
================================================================================
'''


# 每天交易时要做的事情
def func(context):
    if g.if_trade == True:
        # 获得调仓日的日期字符串
        end = context.previous_date
        ais, market = get_df(context, end)  # 当前因子值
        ais = ais.sort_index(by='cap', ascending=True)
        ais = ais.head(1)
        print(ais)

        stock_sort = ais.index
        # 需要买入的
        stock_buy = stock_sort
        # stock_buy=list(stock_sort)-list(context.portfolio.positions)  .['000001.XSHE'].avg_cost
        # 卖出

        # print(stock_buy,context.portfolio.positions.values)
        if len(context.portfolio.positions) > 0:
            for i in context.portfolio.positions.values():
                code, price, avg_cost = i.security, i.price, i.avg_cost
            print(code)
            MaxDrawdown = get_MaxDrawdown(code, end)

            change = price / avg_cost - 1
            if market < 0 and change < -0.196:
                order_stock_sell(context, stock_sort)
            if market > 0 and change < -0.19:
                order_stock_sell(context, stock_sort)
            if market < 0 and change > 0.2 and MaxDrawdown < -0.05505:
                order_stock_sell(context, stock_sort)
            if market > 0 and change > 0.22 and MaxDrawdown < -0.0685:
                order_stock_sell(context, stock_sort)
        # 总资产
        g.everyStock = context.portfolio.total_value
        # 买入

        for i in stock_buy:
            order_stock_buy(context, stock_buy)
    g.if_trade = False


# 6
# 获得卖出信号，并执行卖出操作
def order_stock_sell(context, stock_sort):
    # 对于不需要持仓的股票，全仓卖出
    for stock in context.portfolio.positions:
        # 除去排名前g.N个股票（选股！）
        if stock not in stock_sort:
            stock_sell = stock
            order_target_value(stock_sell, 0)


# 7
# 获得买入信号，并执行买入操作
# 输入：context, data，已排序股票列表stock_sort-list类型
def order_stock_buy(context, stock_sort):
    # 对于需要持仓的股票，按分配到的份额买入
    # 如果已经在持仓里，则不再运行
    for stock in stock_sort:
        stock_buy = stock
        order_target_value(stock_buy, g.everyStock)


# 获取n年前的今天
def get_n_year_date(context, n):
    now = context.previous_date  # context.current_dt
    last_one_year = int(now.year) - n
    now_date = now.strftime("%Y-%m-%d")[-6:]
    if now_date == '-02-29':
        now_date = '-02-28'
    last_year_date = str(last_one_year) + now_date
    return last_year_date


# 给定日期获取因子数据，并处理
def get_df(context, end):
    days_before = shift_trading_day(end, shift=-84)
    year1 = get_n_year_date(context, 1)
    q = query(valuation.code, valuation.market_cap,
              1 / valuation.pe_ratio,
              ).filter(valuation.code.in_(g.all_stocks))
    df = get_fundamentals(q, date=end)
    df.columns = ['code', 'cap', 'EP']
    df.index = df.code.values
    del df['code']
    df = df[df['cap'] < 55]  # 市值小于55亿
    g.all_stocks = list(df.index)

    # FORCAST=get_FORCAST(g.all_stocks,end)
    # df=pd.merge(FORCAST,df,how='left',left_index=True, right_index=True)
    g.all_stocks = list(df.index)

    df['maxmin'] = get_Maxmin(g.all_stocks, end)
    df = df[df['maxmin'] < 0.65]  # 30天区间涨跌幅
    g.all_stocks = list(df.index)

    if len(g.all_stocks) > 0:
        df['mtms'] = get_df_mtms(g.all_stocks, end)
        df = df.sort_index(by='mtms', ascending=False)
        df = df.head(int(len(df) * 0.5))
        df['cap'] = df['cap'] - df['mtms'] * 200
    # (n+61)天前日期
    days_before = shift_trading_day(end, shift=-(60))
    # (n+61)天前价格
    price = get_price('000300.XSHG', start_date=days_before,
                      end_date=end, frequency='daily', fields='close')
    close = price['close']
    # print(close)
    MACD, MACDsignal, MACDhist = talib.MACD(close,
                                            fastperiod=7, slowperiod=28, signalperiod=5)
    market = MACDhist[-1]

    # print(df)
    # 数据清洗
    df = df.fillna(0)  # nan值替换为0
    df = df.sort_index(by='cap', ascending=True)

    return df, market


# 4
# 某一日的前shift个交易日日期
# 输入：date为datetime.date对象(是一个date，而不是datetime)；shift为int类型
# 输出：datetime.date对象(是一个date，而不是datetime)
def shift_trading_day(date, shift):
    # 获取所有的交易日，返回一个包含所有交易日的 list,元素值为 datetime.date 类型.
    tradingday = get_all_trade_days()
    # 得到date之后shift天那一天在列表中的行标号 返回一个数
    shiftday_index = list(tradingday).index(date) + shift
    # 根据行号返回该日日期 为datetime.date类型
    return tradingday[shiftday_index]


# 过去3天涨幅
def get_df_MTM3(stock_list, end):
    # (n+61)天前日期
    days_before2 = shift_trading_day(end, shift=-21)
    days_before3 = shift_trading_day(end, shift=-42)
    days_before4 = shift_trading_day(end, shift=-84)
    # (n+61)天前价格
    days_price1 = get_price(list(stock_list), start_date=end,
                            end_date=end, frequency='daily', fields='close')['close'].T
    days_price2 = get_price(list(stock_list), start_date=days_before2,
                            end_date=days_before2, frequency='daily', fields='close')['close'].T
    days_price3 = get_price(list(stock_list), start_date=days_before3,
                            end_date=days_before3, frequency='daily', fields='close')['close'].T
    days_price4 = get_price(list(stock_list), start_date=days_before4,
                            end_date=days_before4, frequency='daily', fields='close')['close'].T
    #
    mtm = pd.DataFrame(index=stock_list, columns=[])
    mtm['mtm1'] = days_price1.iloc[:, 0] - days_price2.iloc[:, 0]
    mtm['mtm2'] = days_price1.iloc[:, 0] - days_price2.iloc[:, 0]
    mtm['mtm4'] = days_price1.iloc[:, 0] - days_price2.iloc[:, 0]

    # print(mtm)
    mtm['mtm'] = (mtm['mtm1'] + mtm['mtm2'] + mtm['mtm4']) / 3
    return mtm['mtm']


# 过去3天涨幅
def get_df_mtms(stock_list, end):
    # (n+61)天前日期
    days_before2 = shift_trading_day(end, shift=2)

    # (n+61)天前价格
    days_price1 = get_price(list(stock_list), start_date=end,
                            end_date=end, frequency='daily', fields='close')['close'].T
    days_price2 = get_price(list(stock_list), start_date=end,
                            end_date=days_before2, frequency='daily', fields='close')['close'].T

    # print(1111111,days_price2)
    mtm = pd.DataFrame(index=stock_list, columns=[])
    mtm['mtms'] = days_price2.iloc[:, -1] / days_price1.iloc[:, 0]
    # print(mtm)
    return mtm['mtms']


def get_Maxmin(stock_list, end):
    # (n+61)天前日期
    days_before = shift_trading_day(end, shift=-(21))
    # (n+61)天前价格
    days_price2 = get_price(list(stock_list), start_date=days_before,
                            end_date=end, frequency='daily', fields='close')['close'].T
    Maxmin = pd.DataFrame(index=stock_list, columns=[])
    Maxmin['max'] = days_price2.max(axis=1)
    Maxmin['min'] = days_price2.min(axis=1)
    Maxmin['Maxmin'] = (Maxmin['max'] / Maxmin['min']) - 1
    # min_price['net_profit_ttm0']=days_price1.iloc[:,0]-days_price2.iloc[:,0]
    return Maxmin['Maxmin']


# 业绩预告
def get_FORCAST(stocks, end):
    days_before = shift_trading_day(end, shift=-84)
    q = query(finance.STK_FIN_FORCAST).filter(finance.STK_FIN_FORCAST.code.in_(stocks),
                                              finance.STK_FIN_FORCAST.pub_date >= days_before,
                                              finance.STK_FIN_FORCAST.pub_date <= end
                                              ).limit(3000)
    df1 = finance.run_query(q)
    # print(df1)
    df1 = df1.sort_index(by='end_date', ascending=False)
    a = df1['end_date'].values[0]
    df1 = df1[df1['end_date'] == a]
    df1 = df1[df1['profit_min'] > 0]

    df1 = df1.drop_duplicates(subset='code')
    df1.index = df1['code']

    stocks = list(df1['code'])
    # print(df1['profit_min'])
    # 按end 获取已发布业绩
    head = str(a)[:4]
    middle = str(a)[5:7]
    # mm=datetime.date(a)
    # print(a,head,middle)

    if middle == '03':
        k = 1
    if middle == '06':
        k = 2
    if middle == '09':
        k = 3
    if middle == '12':
        k = 4

    if k > 1:
        q = query(income.code, income.net_profit,
                  ).filter(valuation.code.in_(stocks))
        rets = [get_fundamentals(q, statDate=head + 'q' + str(i)) for i in range(1, k)]
        # actual=rets[0]
        # print(rets[0])
        actual = pd.DataFrame(index=rets[0].index, columns=['net_profit'])
        actual['net_profit'] = 0
        # actual.reindex(actual['code'])  rets[0]['code'

        for i in range(0, k - 1):
            # print(rets[i])
            actual['net_profit'] = actual['net_profit'] + rets[i]['net_profit']
        actual.index = rets[0]['code']
        # print(actual)
    if k == 1:
        actual = pd.DataFrame(index=stocks, columns=['net_profit'])
        actual['net_profit'] = 0
    df1['net_profit'] = df1['profit_min'] + df1['profit_min'] - actual['net_profit']
    # print(df1['net_profit'])

    last = datetime.datetime.strptime(head, '%Y')
    last_year_date = str(int(last.year) - 1) + str(a)[4:]

    q = query(income.code, income.net_profit, income.statDate
              ).filter(income.statDate == last_year_date, valuation.code.in_(stocks))
    p0 = get_fundamentals(q, statDate=str(int(last.year) - 1) + 'q' + str(k))
    p0.index = p0.code.values
    # print(p0['net_profit'])
    p0['inc'] = df1['net_profit'] / p0['net_profit'] - 1
    # print(p0['inc'])
    return p0[['inc', 'net_profit']]


def get_MaxDrawdown(stock_list, end):
    # (n+61)天前日期
    days_before = shift_trading_day(end, shift=-(66))
    # (n+61)天前价格
    days_price2 = get_price(stock_list, start_date=days_before,
                            end_date=end, frequency='daily', fields='close')['close'].T
    # print(days_price2)

    max1 = days_price2.max()
    MaxDrawdown = (days_price2[-1] / max1) - 1
    # min_price['net_profit_ttm0']=days_price1.iloc[:,0]-days_price2.iloc[:,0]
    return MaxDrawdown


'''
================================================================================
每天收盘后
================================================================================
'''


# 每天收盘后要做的事情
def after_trading_end(context):
    return