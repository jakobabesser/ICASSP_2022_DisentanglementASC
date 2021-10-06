"""
define experimental configurations (see Fig. 2 in the paper)
  "do_masking": if True, embedding masking is applied and embedding size is doubled (as we effectively only use half
                of it then for learning ASC so that this is comparable to the no-masking settings)
  "use_td": Switch whether to use target domain data (or just source domain data) for training
  "use_td_targets": Switch whether to use target domain ASC labels (supervised) or not (unsupervised)
  "asc_and_dc": Switch whether to compute losses for both ASC and domain classification (True) or just ASC (False)
  "do_mezza": Apply band-wise statistics matching between target domain test data and source domain training data
              as proposed by Mezza et al. (EUSIPCO 2021)
"""
configs = [{'label': 'C0', 'do_masking': False, 'use_td': False,
            'use_td_targets': False, 'asc_and_dc': False, "do_mezza": False},
           {'label': 'C0-mezza', 'do_masking': False, 'use_td': False,
            'use_td_targets': False, 'asc_and_dc': False, "do_mezza": True},
           {'label': 'C1', 'do_masking': False, 'use_td': True,
            'use_td_targets': True, 'asc_and_dc': False, "do_mezza": False},
           {'label': 'C2', 'do_masking': False, 'use_td': True,
            'use_td_targets': False, 'asc_and_dc': True, "do_mezza": False},
           {'label': 'C2M', 'do_masking': True, 'use_td': True,
            'use_td_targets': False, 'asc_and_dc': True, "do_mezza": False},
           {'label': 'C3', 'do_masking': False, 'use_td': True,
            'use_td_targets': True, 'asc_and_dc': True, "do_mezza": False},
           {'label': 'C3M', 'do_masking': True, 'use_td': True,
            'use_td_targets': True, 'asc_and_dc': True, "do_mezza": False}]