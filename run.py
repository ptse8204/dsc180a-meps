# This file would run the notebook output when given arg == "main"

import sys
import json

sys.path.insert(0, 'src')
import eda
import Model_Dev_No_Debias
import add_model_dev
import reweighing_LR
import reweighing_RF
# Section 6 result summary would be on main
import explain


def main():
    eda.main()
    Model_Dev_No_Debias.main()
    add_model_dev.main()
    reweighing_LR.main()
    reweighing_RF.main()
    explain.main()
    
#     data_config = json.load(open('config/data-params.json'))
#     eda_config = json.load(open('config/eda-params.json'))

#     if 'data' in targets:

#         data = generate_data(**data_config)
#         save_data(data, **data_config)

#     if 'eda' in targets:

#         try:
#             data
#         except NameError:
#             data = pd.read_csv(data_config['data_fp'])

#         generate_stats(data, **eda_config)
        
#         # execute notebook / convert to html
#         convert_notebook(**eda_config)


 if __name__ == '__main__':

     targets = sys.argv[1:]
     main()
