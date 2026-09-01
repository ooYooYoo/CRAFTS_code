import torch.utils.data as data
import pandas as pd
import random


class CRAFTSDataset(data.Dataset):
    def __init__(self, data_dir=None, stage='test', cans_num=20, sep=", ", arlib_dataset=None, candidates_map=None):
        self.arlib_dataset = arlib_dataset
        self.stage = stage
        
        self.cans_num = cans_num
        self.sep = sep
        
        self.candidates_map = candidates_map if candidates_map is not None else {}

        if self.arlib_dataset is None:
            raise ValueError("In 'ARLib-only' mode, arlib_dataset (DataLoader) must be provided.")

        
        self.item_id2meta = self._get_meta_from_arlib()

        
        self.session_data = self._build_session_from_arlib()

    def __len__(self):
        return len(self.session_data)

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]

        
        u_int = temp['user_id']

        
        u_raw = str(self.arlib_dataset.id2user[u_int])

        candidates_internal = []

        if u_raw in self.candidates_map:
            cands_raw = self.candidates_map[u_raw]

            # Raw Item ID -> Internal Item ID
            for c_raw in cands_raw:
                c_int = self.arlib_dataset.item.get(str(c_raw))
                if c_int is None:
                    c_int = self.arlib_dataset.item.get(int(c_raw)) if str(c_raw).isdigit() else None
                if c_int is not None:
                    candidates_internal.append(c_int)

        target_item = temp['next']
        min_candidates = 20

        
        if target_item not in candidates_internal:
            candidates_internal.append(target_item)

        
        candidates_internal = list(dict.fromkeys(candidates_internal))

        
        if self.stage != 'test' and len(candidates_internal) < min_candidates:
            all_items = list(self.arlib_dataset.id2item.keys())
            existing = set(candidates_internal)

            neg_pool = [x for x in all_items if x not in existing and x != target_item]
            random.shuffle(neg_pool)

            need = min_candidates - len(candidates_internal)
            candidates_internal.extend(neg_pool[:need])

        
        candidates_internal = list(dict.fromkeys(candidates_internal))
        random.shuffle(candidates_internal)


        cans_name = [self.item_id2meta.get(can, f"Item_{can}") for can in candidates_internal]
        seq_names = [self.item_id2meta.get(x, f"Item_{x}") for x in temp['seq_unpad']]
        target_name = self.item_id2meta.get(temp['next'], f"Item_{temp['next']}")

        return {
            'user_id': u_int,                 
            'seq': temp['seq_unpad'],         
            'len_seq': len(temp['seq_unpad']),
            'seq_str': self.sep.join(seq_names),  
            'cans': candidates_internal,      
            'len_cans': len(candidates_internal), 
            'cans_str': self.sep.join(cans_name),  
            'next': temp['next'],
            'ground_truth_id': temp['next'],
            'ground_truth': target_name,
            'next_name': target_name,
        }


    def _get_meta_from_arlib(self):
        meta_dict = {}
        for internal_id, raw_id in self.arlib_dataset.id2item.items():
            meta = self.arlib_dataset.get_item_meta(internal_id)
            if meta and 'title' in meta:
                meta_dict[internal_id] = str(meta['title'])
            else:
                meta_dict[internal_id] = str(raw_id)
        return meta_dict

    def _build_session_from_arlib(self):
        data_list = []

        if self.stage == 'test':
            source = self.arlib_dataset.test_data
        elif self.stage == 'val':
            source = self.arlib_dataset.val_data
        else:
            source = self.arlib_dataset.training_data

        for entry in source:
            u_int, i_int = entry[0], entry[1]
            history = entry[4]

            history = history[-50:]
            if not history:
                continue

            data_list.append({
                'user_id': u_int,
                'seq_unpad': history,
                'next': i_int
            })

        print(f"[Dataset] Built from ARLib {self.stage} set. Total: {len(data_list)}")
        return pd.DataFrame(data_list)
