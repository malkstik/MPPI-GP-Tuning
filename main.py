from run import *

# vanillaHP = [0.5, 0.01, 1., 1., 0.1]
# evalHP(vanillaHP, 100, 4)

hyperparameters = [2.6514342538430498, 1.6081216502910927, 0.9409802501553195, 2.3841914681416014, 0.105511167334764]
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
