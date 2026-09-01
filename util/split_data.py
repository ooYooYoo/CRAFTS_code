import os
import json
import random
import collections
import importlib


class BenchmarkSplitter:
    

    def __init__(self, raw_loader, agent_args, rec_args, dataset, agentname, save_dir):
        self.raw_loader = raw_loader          # DataLoader_recbole_Raw(recommend_args)
        self.agent_args = agent_args          # agent_parse_args()
        self.rec_args = rec_args              # recommend_parse_args()

        self.dataset_dir = raw_loader.dataset_dir
        self.agentname = agentname
        self.save_dir = save_dir
        self.dataset = dataset

        ds = dataset
        self.train_path = os.path.join(self.dataset_dir, f"{ds}.train.inter")
        self.valid_path = os.path.join(self.dataset_dir, f"{ds}.valid.inter")
        self.test_path = os.path.join(self.dataset_dir, f"{ds}.test.inter")

        self.cand_dir = os.path.join(self.save_dir, self.agentname, self.dataset)
        self.candidates_path = os.path.join(self.cand_dir, "candidates.json")

        
        self.field_sep = getattr(self.rec_args, "field_separator", "\t")
        self.seq_sep = getattr(self.rec_args, "seq_separator", " ")

        self.USER_ID_FIELD = getattr(self.rec_args, "USER_ID_FIELD", "user_id")
        self.ITEM_ID_FIELD = getattr(self.rec_args, "ITEM_ID_FIELD", "item_id")
        self.RATING_FIELD = getattr(self.rec_args, "RATING_FIELD", "rating")
        self.TIME_FIELD = getattr(self.rec_args, "TIME_FIELD", "timestamp")

        self.max_hist_len = int(getattr(self.rec_args, "MAX_ITEM_LIST_LENGTH", 50))
        self.SEQ_FIELD = f"{self.ITEM_ID_FIELD}_list"

        self.cand_num = 20
        self.seed = int(getattr(self.rec_args, "seed", 2025))
        self.candidate_generate_mode = str(
            getattr(self.rec_args, "candidate_generate_mode", "random")
        ).strip().lower()

    def _benchmark_files_exist(self):
        
        return all(os.path.isfile(p) for p in [self.train_path, self.valid_path, self.test_path])

    @staticmethod
    def _parse_user_limit(raw):
        if raw is None:
            return None
        s = str(raw).strip().lower()
        if s in ("", "all", "full", "none", "-1", "0"):
            return None
        try:
            n = int(float(s))
            return n if n > 0 else None
        except Exception:
            return None

    def split_and_save(self, overwrite=None, generate_candidates=True):
        files_exist = self._benchmark_files_exist()
        force_split = bool(getattr(self.rec_args, "split_data", False))
        need_split = (not files_exist) or force_split

        if not need_split:
            print("[BenchmarkSplitter] Benchmark files already exist and split_data=False, skip splitting.")
            return

        reason = "missing benchmark files" if (not files_exist) else "split_data=True"
        print(f"[BenchmarkSplitter] Splitting triggered ({reason}).")

        filter_num = int(getattr(self.agent_args, "MIN_ITEM_LIST_LENGTH", 5))
        if filter_num < 3:
            if filter_num > 0:
                print(f"[BenchmarkSplitter] Warning: filter_num={filter_num} is too small for LOO. Auto-adjusted to 3.")
            filter_num = 3

        
        val_raw = getattr(self.agent_args, "val_user_num", getattr(self.rec_args, "val_user_num", "all"))
        test_raw = getattr(self.agent_args, "test_user_num", getattr(self.rec_args, "test_user_num", "all"))
        val_num = self._parse_user_limit(val_raw)
        test_num = self._parse_user_limit(test_raw)

        print(f"[BenchmarkSplitter] Filter Threshold: Users with < {filter_num} interactions will be dropped.")
        if val_num is None:
            print("[BenchmarkSplitter] Val User Limit: all")
        else:
            print(f"[BenchmarkSplitter] Val User Limit: {val_num}")
        if test_num is not None:
            print(f"[BenchmarkSplitter] Test User Limit: {test_num}")
        else:
            print("[BenchmarkSplitter] Test User Limit: all")

        data = self.raw_loader.inter_data
        if not data:
            raise RuntimeError("[BenchmarkSplitter] No interaction data loaded from raw .inter")

        user_interactions = collections.defaultdict(list)
        for (u, i, r, t) in data:
            user_interactions[u].append((u, i, float(r), float(t)))

        
        rng = random.Random(self.seed)
        eligible_users = [u for u, inters in user_interactions.items() if len(inters) >= filter_num]
        total_users = len(user_interactions)
        kept_users = len(eligible_users)
        dropped_users = total_users - kept_users

        users_shuffled = list(eligible_users)
        rng.shuffle(users_shuffled)

        if test_num is None or test_num >= kept_users:
            test_eval_users = set(eligible_users)
        else:
            test_eval_users = set(users_shuffled[:test_num])

        if val_num is None or val_num >= kept_users:
            val_eval_users = set(eligible_users)
        else:
            # Prefer overlap with test users when both limits are finite.
            test_order = [u for u in users_shuffled if u in test_eval_users]
            if len(test_order) >= val_num:
                val_eval_users = set(test_order[:val_num])
            else:
                val_eval_users = set(test_order)
                need = val_num - len(test_order)
                remain = [u for u in users_shuffled if u not in test_eval_users]
                val_eval_users.update(remain[:need])
                print(
                    f"[BenchmarkSplitter] Info: val({val_num}) > test overlap pool({len(test_order)}); "
                    f"filled remaining {max(0, need)} users from non-test set."
                )

        train_rows, valid_rows, test_rows = [], [], []

        
        for u, inters in user_interactions.items():
            if len(inters) < filter_num:
                continue

            inters.sort(key=lambda x: x[3])  # timestamp asc
            seq_items = [x[1] for x in inters]

            test_idx = len(inters) - 1
            valid_idx = len(inters) - 2
            train_last_target_idx = len(inters) - 3

            
            for k in range(1, train_last_target_idx + 1):
                target = inters[k]
                hist = seq_items[:k]
                if self.max_hist_len > 0:
                    hist = hist[-self.max_hist_len:]
                if len(hist) == 0:
                    continue
                train_rows.append((u, target[1], target[2], target[3], hist))

            if u in val_eval_users:
                valid_target = inters[valid_idx]
                valid_hist = seq_items[:valid_idx]
                if self.max_hist_len > 0:
                    valid_hist = valid_hist[-self.max_hist_len:]
                valid_rows.append((u, valid_target[1], valid_target[2], valid_target[3], valid_hist))

            if u in test_eval_users:
                test_target = inters[test_idx]
                test_hist = seq_items[:test_idx]
                if self.max_hist_len > 0:
                    test_hist = test_hist[-self.max_hist_len:]
                test_rows.append((u, test_target[1], test_target[2], test_target[3], test_hist))

        print(f"[BenchmarkSplitter] Total Users: {total_users}")
        print(f"[BenchmarkSplitter] Filtered (<{filter_num}): {dropped_users} users dropped.")
        print(f"[BenchmarkSplitter] Kept: {kept_users} users.")
        overlap_users = len(val_eval_users.intersection(test_eval_users))
        print(f"[BenchmarkSplitter] Eval Users (valid): {len(val_eval_users)}")
        print(f"[BenchmarkSplitter] Eval Users (test): {len(test_eval_users)}")
        print(f"[BenchmarkSplitter] Eval Users Overlap (val∩test): {overlap_users}")
        print(f"[BenchmarkSplitter] Train Samples: {len(train_rows)}, Valid: {len(valid_rows)}, Test: {len(test_rows)}")

        self._save_inter(self.train_path, train_rows)
        self._save_inter(self.valid_path, valid_rows)
        self._save_inter(self.test_path, test_rows)

        if generate_candidates:
            self._save_candidates_json(test_rows, valid_rows, user_interactions)


    def _save_inter(self, path, rows):
        header = [
            f"{self.USER_ID_FIELD}:token",
            f"{self.ITEM_ID_FIELD}:token",
            f"{self.RATING_FIELD}:float",
            f"{self.TIME_FIELD}:float",
            f"{self.SEQ_FIELD}:token_seq",
        ]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.field_sep.join(header) + "\n")
            for (u, i, r, t, hist) in rows:
                hist_str = self.seq_sep.join(str(x) for x in hist)
                f.write(self.field_sep.join([str(u), str(i), str(float(r)), str(float(t)), hist_str]) + "\n")
        print(f"Saved {path}")

    def _save_candidates_json(self, test_rows, valid_rows, user_interactions):
        mode = self.candidate_generate_mode
        if mode == "tool":
            self._save_candidates_json_by_tool(
                test_rows, valid_rows, user_interactions, shuffle_candidates=False
            )
        elif mode == "tool_shuffle":
            self._save_candidates_json_by_tool(
                test_rows, valid_rows, user_interactions, shuffle_candidates=True
            )
        else:
            if mode != "random":
                print(f"[BenchmarkSplitter] Unknown candidate_generate_mode='{mode}', fallback to random.")
            self._save_candidates_json_random(test_rows, valid_rows, user_interactions)

    def _save_candidates_json_random(self, test_rows, valid_rows, user_interactions):
        cand_num = max(1, int(self.cand_num))
        rng = random.Random(self.seed)

        os.makedirs(self.cand_dir, exist_ok=True)
        test_path = self.candidates_path
        val_path = os.path.join(self.cand_dir, "candidates_val.json")

        all_items_set = set()
        for _, inters in user_interactions.items():
            for (_, i, _, _) in inters:
                all_items_set.add(str(i))
        all_items_list = list(all_items_set)
        n_items = len(all_items_list)

        gt_test_by_user = {str(u): str(i) for (u, i, r, t, hist) in test_rows}
        gt_val_by_user = {str(u): str(i) for (u, i, r, t, hist) in valid_rows}

        def build_candidates_for_users(gt_by_user, extra_forbidden_by_user=None):
            candidates = {}

            for u, inters in user_interactions.items():
                u_str = str(u)
                if u_str not in gt_by_user:
                    continue

                gt_item = gt_by_user[u_str]
                forbidden = set(str(x[1]) for x in inters)
                forbidden.add(gt_item)

                if extra_forbidden_by_user is not None and u_str in extra_forbidden_by_user:
                    forbidden |= set(extra_forbidden_by_user[u_str])

                needed = cand_num - 1
                negs = []

                if n_items == 0:
                    candidates[u_str] = [gt_item]
                    continue

                available_cnt_est = n_items - len(forbidden)
                if available_cnt_est <= 0:
                    candidates[u_str] = [gt_item]
                    continue

                if needed > n_items * 0.8:
                    pool = [x for x in all_items_list if x not in forbidden]
                    negs = rng.sample(pool, min(needed, len(pool)))
                else:
                    max_try = 50
                    tries = 0
                    while len(negs) < needed and tries < max_try:
                        tries += 1
                        batch_size = min((needed - len(negs)) * 2, n_items)
                        potential = rng.sample(all_items_list, batch_size)
                        for item in potential:
                            if item not in forbidden and item not in negs:
                                negs.append(item)
                                if len(negs) == needed:
                                    break

                    if len(negs) < needed:
                        pool = [x for x in all_items_list if x not in forbidden and x not in negs]
                        if pool:
                            fill = rng.sample(pool, min(needed - len(negs), len(pool)))
                            negs.extend(fill)

                cand_list = [gt_item] + negs
                rng.shuffle(cand_list)
                candidates[u_str] = cand_list

            return candidates

        test_candidates = build_candidates_for_users(gt_test_by_user, extra_forbidden_by_user=None)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_candidates, f, ensure_ascii=False)
        print(f"[BenchmarkSplitter] Saved test candidates -> {test_path}")
        print(f"[BenchmarkSplitter] Stats(TEST): Users={len(test_candidates)}, Cand_Num={cand_num}, Mode=random")

        extra_forbidden = {u: set(items) for u, items in test_candidates.items()}
        val_candidates = build_candidates_for_users(gt_val_by_user, extra_forbidden_by_user=extra_forbidden)
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_candidates, f, ensure_ascii=False)
        print(f"[BenchmarkSplitter] Saved val candidates -> {val_path}")
        print(f"[BenchmarkSplitter] Stats(VAL): Users={len(val_candidates)}, Cand_Num={cand_num}, Mode=random")

    def _build_tool_model(self):
        from util.DataLoader import DataLoader_recbole

        data_loader = DataLoader_recbole(self.rec_args)
        tool_name = getattr(self.agent_args, "tool_name", "SASRec")
        tool_type = getattr(self.agent_args, "tool_type", "sequential")
        module = importlib.import_module(f"recommender.{tool_type}.{tool_name}")
        ToolClass = getattr(module, tool_name)

        if getattr(self.agent_args, "config_file_path", None):
            config_file_list = [self.agent_args.config_file_path]
        else:
            default_conf = os.path.join("tool_conf", self.agentname, self.dataset, f"{tool_name}.yaml")
            if os.path.exists(default_conf):
                config_file_list = [default_conf]
            else:
                config_file_list = [os.path.join("tool_conf", "default", f"{tool_name}.yaml")]

        tool_dir = os.path.join(self.save_dir, self.agentname, self.dataset, "clean", "tool")
        os.makedirs(tool_dir, exist_ok=True)
        tool_ckpt = os.path.join(tool_dir, f"{tool_name}_model.pth")

        tool_instance = ToolClass(self.rec_args, data_loader, config_file_list, tool_dir, self.agent_args)
        if not os.path.exists(tool_ckpt):
            print(f"[BenchmarkSplitter] Tool checkpoint not found, training {tool_name} -> {tool_ckpt}")
            tool_instance.train(tool_ckpt)
        else:
            print(f"[BenchmarkSplitter] Using existing tool checkpoint -> {tool_ckpt}")

        tool_instance.arlib_dataset = data_loader
        loaded_tool = tool_instance.load(tool_ckpt)
        return loaded_tool, data_loader

    def _save_candidates_json_by_tool(self, test_rows, valid_rows, user_interactions, shuffle_candidates=False):
        cand_num = max(1, int(self.cand_num))
        rng = random.Random(self.seed)

        os.makedirs(self.cand_dir, exist_ok=True)
        test_path = self.candidates_path
        val_path = os.path.join(self.cand_dir, "candidates_val.json")

        gt_test_by_user = {str(u): str(i) for (u, i, r, t, hist) in test_rows}
        gt_val_by_user = {str(u): str(i) for (u, i, r, t, hist) in valid_rows}
        hist_test_by_user = {str(u): [str(x) for x in hist] for (u, i, r, t, hist) in test_rows}
        hist_val_by_user = {str(u): [str(x) for x in hist] for (u, i, r, t, hist) in valid_rows}

        all_items_set = set()
        for _, inters in user_interactions.items():
            for (_, i, _, _) in inters:
                all_items_set.add(str(i))
        all_items_list = list(all_items_set)

        tool_model, data_loader = self._build_tool_model()

        def fill_from_pool(forbidden, need):
            if need <= 0:
                return []
            pool = [x for x in all_items_list if x not in forbidden]
            if not pool:
                return []
            return rng.sample(pool, min(need, len(pool)))

        def build_candidates(gt_by_user, hist_by_user, extra_forbidden_by_user=None, forbid_val_item_in_test=False):
            out = {}
            overlap_gt_count = 0
            for u_str, gt_item in gt_by_user.items():
                hist_raw = hist_by_user.get(u_str, [])
                hist_int = [data_loader.item[h] for h in hist_raw if h in data_loader.item]

                forbidden = set(hist_raw)
                forbidden.add(gt_item)
                if extra_forbidden_by_user is not None and u_str in extra_forbidden_by_user:
                    forbidden |= set(extra_forbidden_by_user[u_str])
                if forbid_val_item_in_test and u_str in gt_val_by_user:
                    forbidden.add(gt_val_by_user[u_str])

                exclude_items = list(forbidden)
                tool_k = max(cand_num * 10, 200)
                recs = tool_model.recommend_topk_all_items(
                    item_sequence=hist_int,
                    k=tool_k,
                    filter_seen=True,
                    exclude_items=exclude_items,
                    with_scores=False,
                    with_names=False,
                )

                negs = []
                for rec in recs:
                    item_raw = str(rec.get("item_raw"))
                    if item_raw in forbidden or item_raw in negs:
                        continue
                    negs.append(item_raw)
                    if len(negs) >= cand_num - 1:
                        break

                if len(negs) < cand_num - 1:
                    fill = fill_from_pool(forbidden | set(negs), cand_num - 1 - len(negs))
                    negs.extend(fill)

                cand = [gt_item] + negs[: max(0, cand_num - 1)]
                out[u_str] = cand

                if extra_forbidden_by_user is not None and u_str in extra_forbidden_by_user:
                    overlap = set(cand) & set(extra_forbidden_by_user[u_str])
                    if overlap:
                        if overlap == {gt_item}:
                            overlap_gt_count += 1
                        else:
                            keep = [gt_item]
                            kept = set(keep)
                            for x in cand[1:]:
                                if x in extra_forbidden_by_user[u_str] or x in kept:
                                    continue
                                keep.append(x)
                                kept.add(x)
                            if len(keep) < cand_num:
                                refill_forbidden = forbidden | set(kept)
                                refill = fill_from_pool(refill_forbidden, cand_num - len(keep))
                                keep.extend(refill)
                            out[u_str] = keep[:cand_num]

            if overlap_gt_count > 0:
                print(
                    f"[BenchmarkSplitter] Warning: {overlap_gt_count} users have val GT overlapping test candidates."
                )
            return out

        mode_name = "tool_shuffle" if shuffle_candidates else "tool"

        def shuffle_candidate_dict(candidates):
            if not shuffle_candidates:
                return
            for cand_list in candidates.values():
                if len(cand_list) > 1:
                    rng.shuffle(cand_list)

        print("[BenchmarkSplitter] Candidate mode=tool, building test candidates with tool scores...")
        test_candidates = build_candidates(
            gt_by_user=gt_test_by_user,
            hist_by_user=hist_test_by_user,
            extra_forbidden_by_user=None,
            forbid_val_item_in_test=True,
        )
        shuffle_candidate_dict(test_candidates)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_candidates, f, ensure_ascii=False)
        print(f"[BenchmarkSplitter] Saved test candidates -> {test_path}")
        print(f"[BenchmarkSplitter] Stats(TEST): Users={len(test_candidates)}, Cand_Num={cand_num}, Mode={mode_name}")

        print("[BenchmarkSplitter] Building val candidates with no overlap against test candidates...")
        extra_forbidden = {u: set(items) for u, items in test_candidates.items()}
        val_candidates = build_candidates(
            gt_by_user=gt_val_by_user,
            hist_by_user=hist_val_by_user,
            extra_forbidden_by_user=extra_forbidden,
            forbid_val_item_in_test=False,
        )
        shuffle_candidate_dict(val_candidates)
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_candidates, f, ensure_ascii=False)
        print(f"[BenchmarkSplitter] Saved val candidates -> {val_path}")
        print(f"[BenchmarkSplitter] Stats(VAL): Users={len(val_candidates)}, Cand_Num={cand_num}, Mode={mode_name}")
