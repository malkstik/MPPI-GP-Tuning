from run import *

# vanillaHP = [0.5, 0.01, 1., 1., 0.1]
# evalHP(vanillaHP, 100, 4)

hyperparameters = [5.38209705132172, 0.000818226196113964, 5.69249384639353, 3.435367933700074 6.2953055533397855]
evalHP(hyperparameters, 100, 4)

# colData = True
# trainGP = True
# OBS_INIT = 4
# TS = True
# if colData:
#     collected_data = collect_data(OBS_INIT)
# train_x, train_y = load_data(OBS_INIT)
# if trainGP:
#     train_gp(train_x, train_y, OBS_INIT)

# if TS:
#     optimum_hp, optimum_cost = run_TS(train_x, train_y, OBS_INIT)
#     print('Optimal HP: ', optimum_hp)
#     print('Optimal Cost: ', optimum_cost)
# else:
#     run_CMA(OBS_INIT)
