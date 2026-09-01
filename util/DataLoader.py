import numpy as np
from collections import defaultdict
from util.FileIO import FileIO
import scipy.sparse as sp
from re import split
import random
import os
import json



class DataLoader():
    def __init__(self, args):
        
        self.training_data = FileIO.load_data_set(args.data_path + args.dataset + args.training_data)
        self.val_data = FileIO.load_data_set(args.data_path + args.dataset + args.val_data)
        self.test_data = FileIO.load_data_set(args.data_path + args.dataset + args.test_data)

        self.test_data = self.test_data[:1000]   

        
        self.dataName=args.dataset
        
        
        self.user = {}
        
        self.item = {}
        
        self.id2user = {}
        
        self.id2item = {}
        
        self.training_set_u = defaultdict(dict)
        
        self.training_set_i = defaultdict(dict)
        
        self.val_set = defaultdict(dict)
        
        self.val_set_item = set()
        
        self.test_set = defaultdict(dict)
        
        self.test_set_item = set()
        
        self.__generate_set()
        
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        
        
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        
        self.interaction_mat = self.__create_sparse_interaction_matrix()
        #popularity_item[self.item[u]] = len(self.training_set_i[u])
        
        print("user num: {}, item num: {}, training data size: {}".format(self.user_num, self.item_num, len(self.training_data)))
        print("Density of training data: {:.6f}".format(len(self.training_data) / (self.user_num * self.item_num)))
        
        # Optionally load item metadata if arguments specify or if needed for Agent
        
        if hasattr(args, 'load_meta') and args.load_meta:
            self.load_item_metadata(args)
            print("Item metadata loaded.")

    
    def __generate_set(self):
        
        for entry in self.training_data:
            # Flexible unpacking to handle optional timestamp
            if len(entry) == 4:
                
                
                user, item, rating, timestamp = entry
            else:
                user, item, rating = entry
                timestamp = 0 # Default if missing

            
            if user not in self.user:
                
                #self.user["U123"] = 0
                self.user[user] = len(self.user) 
                
                self.id2user[self.user[user]] = user 
            if item not in self.item:
                self.item[item] = len(self.item) 
                self.id2item[self.item[item]] = item 
                # userList.append
            
            
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
        
        for entry in self.val_data:
            if len(entry) == 4:
                user, item, rating, timestamp = entry
            else:
                 user, item, rating = entry
            
            
            
            if user not in self.user:
                continue
            # [FIX] Register validation items to global item mapping
            
            
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
            self.val_set[user][item] = rating
            self.val_set_item.add(item)
        
        for entry in self.test_data:
            if len(entry) == 4:
                user, item, rating, timestamp = entry
            else:
                 user, item, rating = entry 

            
            if user not in self.user:
                continue
            # [FIX] Register test items to global item mapping
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
            self.test_set[user][item] = rating
            self.test_set_item.add(item)
        print(len(self.test_set), "users to process.")
    
    
    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        
        n_nodes = self.user_num + self.item_num
        
        
        
        #row_idx = [0, 0, 1] 
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        #col_idx = [0, 1, 1]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        
        
        ratings = np.ones_like(user_np, dtype=np.float32)
        
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        
        adj_mat = tmp_adj + tmp_adj.T
        
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    
    def normalize_graph_mat(self, adj_mat):
        
        shape = adj_mat.get_shape()
        
        rowsum = np.array(adj_mat.sum(1))
        
        if shape[0] == shape[1]:
            
            #rowsum = [[2], [1], [0], [2], [1]]
            #d_inv = [0.7071, 1.0, inf, 0.7071, 1.0]
            d_inv = np.power(rowsum, -0.5).flatten()
            
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def load_item_metadata(self, args):
        """
        Load meta data (id2name map) for Agent Recommend.
        """
        self.item_meta_dict = {}
        
        json_path = args.data_path + args.dataset + "/id2meta.json"
        
        if os.path.exists(json_path):
            print(f"Loading item metadata from {json_path}")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta_json = json.load(f)
                    
                    for k, v in meta_json.items():
                        self.item_meta_dict[str(k)] = v
                        try:
                            self.item_meta_dict[int(k)] = v
                        except:
                            pass
            except Exception as e:
                print(f"Error loading JSON metadata: {e}")
        else:
            print(f"Warning: {json_path} not found. Agent will use Item IDs.")

    
    def get_item_name(self, i):
        """
        Get item name by internal item ID.
        """
        if not hasattr(self, 'item_meta_dict'):
             return f"Item_{i}"
             
        if i in self.id2item:
            raw_id = self.id2item[i]
            if raw_id in self.item_meta_dict:
                return self.item_meta_dict[raw_id]
            # Fallback if raw_id is int but stored as str or vice versa
            if str(raw_id) in self.item_meta_dict:
                return self.item_meta_dict[str(raw_id)]
            return str(raw_id)
        return f"Unknown_Item_{i}"

    
    def get_user_history_names(self, u):
        """
        Get the list of item names interacted by user u (internal user ID)
        """
        
        raw_u = self.id2user.get(u)
        if raw_u is None or raw_u not in self.training_set_u:
            return []
        
        item_ids = list(self.training_set_u[raw_u].keys())
        names = []
        for raw_i in item_ids:
            
            if raw_i in self.item:
                
                internal_i = self.item[raw_i]
                names.append(self.get_item_name(internal_i))
            else:
                names.append(str(raw_i))
        return names

    
    def convert_to_laplacian_mat(self, adj_mat):
        
        
        adj_shape = adj_mat.get_shape()
        
        n_nodes = adj_shape[0]+adj_shape[1]
        
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        
        ratings_keep = adj_mat.data
        
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    
    def __create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat
    
    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]
    
    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]
    
    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)
    
    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False
    
    #(["itemA", "itemB", "itemC"], [5, 4, 1])
    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())
    
    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    
    def matrix(self):
        """
        Create and return a sparse interaction matrix with shape (user_num, item_num).
        Used by attack models.
        """
        import scipy.sparse as sp
        row, col, entries = [], [], []
        
        # Build matrix from training_data
        for entry in self.training_data:
            user_internal = int(entry[0])  # Internal user ID (ensure int)
            item_internal = int(entry[1])  # Internal item ID (ensure int)
            rating = float(entry[2]) if len(entry) > 2 else 1.0
            
            row.append(user_internal)
            col.append(item_internal)
            entries.append(rating)
        
        # Determine actual matrix dimensions from max IDs
        
        
        max_user = max(row) + 1 if row else self.user_num
        max_item = max(col) + 1 if col else self.item_num
        
        # Use the larger of declared size or actual max ID
        num_users = max(self.user_num, max_user)
        num_items = max(self.item_num, max_item)
        
        # Create sparse matrix
        interaction_mat = sp.csr_matrix(
            (entries, (row, col)), 
            shape=(num_users, num_items),
            dtype='float32'
        )
        return interaction_mat







def _sort_key_token(x):
    s = str(x)
    return (0, int(s)) if s.isdigit() else (1, s)


class DataLoader_recbole_Raw:
    

    def __init__(self, args):
        self.args = args
        
        self.dataset_dir = os.path.join(args.data_path, args.dataset)
        
        self.inter_path = os.path.join(self.dataset_dir, f"{args.dataset}.inter")

        
        self.field_sep = getattr(args, "field_separator", "\t")

        
        self.USER_ID_FIELD = getattr(args, "USER_ID_FIELD", "user_id")
        self.ITEM_ID_FIELD = getattr(args, "ITEM_ID_FIELD", "item_id")
        self.RATING_FIELD  = getattr(args, "RATING_FIELD", "rating")
        self.TIME_FIELD    = getattr(args, "TIME_FIELD", "timestamp")

        
        
        self.inter_data = self._load_inter_file(self.inter_path)

    def _load_inter_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw inter file not found: {path}")

        data = []
        with open(path, "r", encoding="utf-8") as f:
            
            header = f.readline().rstrip("\n").split(self.field_sep)
            
            fields = [h.split(":")[0] for h in header]

            
            idx_u = fields.index(self.USER_ID_FIELD)
            idx_i = fields.index(self.ITEM_ID_FIELD)
            idx_r = fields.index(self.RATING_FIELD) if self.RATING_FIELD in fields else None
            idx_t = fields.index(self.TIME_FIELD) if self.TIME_FIELD in fields else None

            
            for line in f:
                
                line = line.rstrip("\n")
                if not line:
                    continue
                
                cols = line.split(self.field_sep)

                
                u = cols[idx_u]
                i = cols[idx_i]
                
                r = float(cols[idx_r]) if idx_r is not None and cols[idx_r] != "" else 1.0
                
                t = float(cols[idx_t]) if idx_t is not None and cols[idx_t] != "" else 0.0
                data.append((u, i, r, t))
        return data


class DataLoader_recbole:
    
    
    
    
    def __init__(self, args, agent_args=None):
        
        self.args = args
        
        self.dataName = args.dataset

        
        self.field_sep = getattr(args, "field_separator", "\t")
        
        self.seq_sep = getattr(args, "seq_separator", " ")

        
        self.USER_ID_FIELD = getattr(args, "USER_ID_FIELD", "user_id")
        self.ITEM_ID_FIELD = getattr(args, "ITEM_ID_FIELD", "item_id")
        self.RATING_FIELD = getattr(args, "RATING_FIELD", "rating")
        self.TIME_FIELD = getattr(args, "TIME_FIELD", "timestamp")

        
        self.SEQ_FIELD = f"{self.ITEM_ID_FIELD}_list"  # item_id_list

        
        self.dataset_dir = os.path.join(args.data_path, args.dataset)
        self.train_path = os.path.join(self.dataset_dir, f"{args.dataset}.train.inter")
        self.valid_path = os.path.join(self.dataset_dir, f"{args.dataset}.valid.inter")
        self.test_path  = os.path.join(self.dataset_dir, f"{args.dataset}.test.inter")
        
        self.item_path  = os.path.join(self.dataset_dir, f"{args.dataset}.item")

        
        print(f"[DataLoader] Loading benchmark files from {self.dataset_dir}...")
        train_raw = self._load_benchmark_file(self.train_path)
        valid_raw = self._load_benchmark_file(self.valid_path)
        test_raw  = self._load_benchmark_file(self.test_path)

        
        
        all_users, all_items = set(), set()
        for data_list in [train_raw, valid_raw, test_raw]:
            for (u, i, r, t, hist) in data_list:
                all_users.add(u)       
                all_items.add(i)       
                for hi in hist:
                    all_items.add(hi)  
        
        self.user = {u: idx for idx, u in enumerate(sorted(all_users, key=_sort_key_token))}
        self.item = {i: idx for idx, i in enumerate(sorted(all_items, key=_sort_key_token))}
        
        self.id2user = {idx: u for u, idx in self.user.items()}
        self.id2item = {idx: i for i, idx in self.item.items()}
        
        self.user_num = len(self.user)
        self.item_num = len(self.item)

        
        
        self.training_data = self._to_internal(train_raw)  # (u_int, i_int, r, t, hist_int)
        self.val_data      = self._to_internal(valid_raw)
        self.test_data     = self._to_internal(test_raw)
        
        print(f"[DataLoader] Done. Users: {self.user_num}, Items: {self.item_num}")
        print(f"[DataLoader] Train: {len(self.training_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")

        
        
        self.training_set_u = defaultdict(dict)  # u_int -> {i_int: rating}
        self.training_set_i = defaultdict(dict)  # i_int -> {u_int: rating}
        for (u, i, r, t, hist) in self.training_data:
            self.training_set_u[u][i] = r 
            self.training_set_i[i][u] = r 

        
        
        self._test_hist_seq = {u: hist for (u, i, r, t, hist) in self.test_data}

        
        self.item_meta = {}
        if getattr(args, "load_meta", False):
            self.load_item_metadata(args)

    def _load_benchmark_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        rows = []
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split(self.field_sep)
            fields = [h.split(":")[0] for h in header]

            idx_u = fields.index(self.USER_ID_FIELD)
            idx_i = fields.index(self.ITEM_ID_FIELD)
            idx_r = fields.index(self.RATING_FIELD) if self.RATING_FIELD in fields else None
            idx_t = fields.index(self.TIME_FIELD) if self.TIME_FIELD in fields else None
            
            idx_s = fields.index(self.SEQ_FIELD) if self.SEQ_FIELD in fields else None

            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                cols = line.split(self.field_sep)

                u = cols[idx_u]
                i = cols[idx_i]
                r = float(cols[idx_r]) if idx_r is not None and cols[idx_r] != "" else 1.0
                t = float(cols[idx_t]) if idx_t is not None and cols[idx_t] != "" else 0.0

                hist = []
                if idx_s is not None and idx_s < len(cols):
                    s = cols[idx_s].strip()
                    hist = s.split(self.seq_sep) if s else []

                rows.append((u, i, r, t, hist))
        return rows
    
    
    def _to_internal(self, raw_list):
        out = []
        for (u, i, r, t, hist) in raw_list:
            if u not in self.user or i not in self.item:
                continue
            u_int = self.user[u]
            i_int = self.item[i]
            hist_int = [self.item[hi] for hi in hist if hi in self.item]
            out.append((u_int, i_int, float(r), float(t), hist_int))
        return out

    
    
    def matrix(self):
        row, col, data = [], [], []
        for (u, i, r, t, hist) in self.training_data:
            row.append(u)
            col.append(i)
            data.append(r)
        return sp.csr_matrix((data, (row, col)), shape=(self.user_num, self.item_num))

    
    def get_user_history_by_internal_id(self, u_int):
        return self._test_hist_seq.get(u_int, [])

    def load_item_metadata(self, args):
            self.item_meta = {}
        #     self.item_meta = {
        #     "item1": {
        #         "item_id": "item1",
        #         "title": "The Shawshank Redemption",
        #         "genre": "Drama",
        #         "rating": 9.3
        #     },
        #     "item2": { ... },
        #     ...
        # }
            
            if not os.path.exists(self.item_path):
                print(f"[Warning] Meta file not found: {self.item_path}")
                return

            print(f"[DataLoader] Loading item metadata from {self.item_path}...")

            with open(self.item_path, "r", encoding="utf-8") as f:
                
                try:
                    header_line = f.readline().rstrip("\n")
                    
                    header = header_line.split(self.field_sep)
                except Exception:
                    print("[Error] Failed to read header.")
                    return

                
                fields = []
                types = []
                for h in header:
                    if ":" in h:
                        k, v = h.split(":")[:2]
                        fields.append(k)
                        types.append(v)
                    else:
                        fields.append(h)
                        types.append("token")
                
                
                try:
                    idx_iid = fields.index(self.ITEM_ID_FIELD)
                except ValueError:
                    print(f"[Warning] ID field '{self.ITEM_ID_FIELD}' not found. Skipping metadata.")
                    return

                
                success_count = 0
                error_count = 0
                
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    
                    cols = line.split(self.field_sep)
                    row = {}

                    
                    for idx, (name, ftype) in enumerate(zip(fields, types)):
                        if idx >= len(cols):
                            break
                        
                        val = cols[idx]
                        
                        
                        if ftype == "float" and val:
                            try:
                                row[name] = float(val)
                            except ValueError:
                                
                                
                                row[name] = 0.0 
                                error_count += 1
                        else:
                            row[name] = val
                        # ===================================

                    
                    if idx_iid < len(cols):
                        iid = cols[idx_iid]
                        self.item_meta[str(iid)] = row
                        success_count += 1
            
            print(f"[DataLoader] Metadata loaded. Success: {success_count} items. "
                f"Corrupted Fields (Auto-filled 0.0): {error_count}")
            

    def get_item_meta(self, i_int, fields=None):
        raw_i = str(self.id2item.get(i_int))
        meta = self.item_meta.get(raw_i, {})
        if fields:
            return {k: meta.get(k) for k in fields}
        return meta



class DataLoader_detect():
    def __init__(self, args):
        self.training_data = self.load_data_label(args.detectDatasetPath + '/train.txt')
        self.val_data = self.load_data_label(args.detectDatasetPath + '/val.txt')
        self.test_data = self.load_data_label(args.detectDatasetPath + '/test.txt')
        self.interact_data = self.load_data_interact(args.detectDatasetPath + '/ratings.txt')

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.set_u = defaultdict(dict)
        self.set_i = defaultdict(dict)
        self.__generate_set()
        self.user_num = len(self.set_u)
        self.item_num = len(self.set_i)
        self.interaction_mat = self.__create_sparse_interaction_matrix()

    def __generate_set(self):
        for entry in self.interact_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                # userList.append
            self.set_u[user][item] = rating
            self.set_i[item][user] = rating

    def load_data_interact(self, file):
        data = []
        with open(file) as f:
            for line in f:
                items = split(' ', line.strip())
                user_id = items[0]
                item_id = items[1]
                weight = items[2]
                data.append([user_id, item_id, float(weight)])
        return data

    def load_data_label(self, file):
        data = {}
        with open(file) as f:
            for line in f:
                items = split(' ', line.strip())
                user_id = items[0]
                label = int(items[1])
                data[user_id] = label
        return data

    def __create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.interact_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [pair[2]]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num),dtype=np.float32)
        return interaction_mat

    def shuffle_data(self, data):
        shuffled_keys = list(data.keys())
        random.shuffle(shuffled_keys)
        shuffled_data = {key: data[key] for key in shuffled_keys}
        return shuffled_data
    
    # def expanded_data(self, data):
    #     label_1_samples = {key: value for key, value in data.items() if value == 1}
    #     label_0_samples = {key: value for key, value in data.items() if value == 0}
    #     num = len(label_0_samples) - len(label_1_samples)
    #     existing_label_1_keys = list(label_1_samples.keys())
    #     for i in range(num):
    #         sampled_key = random.choice(existing_label_1_keys)  
    #         new_key = f"{sampled_key}_dup_{i}"
    #         label_1_samples[new_key] = 1

    #     combined_data = {**label_0_samples, **label_1_samples}
