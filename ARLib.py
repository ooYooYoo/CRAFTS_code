import importlib
import logging
import os
from time import localtime, strftime, time

from util.metrics import AgentMetric


class ARLib:
    """Coordinator for the skill-centric ARS pipeline."""

    def __init__(self, recommendModel, recommendArg, agentArg=None):
        self.result = []
        self.recommendModel = recommendModel
        self.recommendModelName = recommendArg.model_name
        self.model_type = recommendArg.model_type
        self.datasetName = recommendArg.dataset
        self.agent = recommendArg.agent
        self.tool_name = agentArg.tool_name if agentArg else None
        self.tool_type = agentArg.tool_type if agentArg else None
        self.recommendArg = recommendArg
        self.top = list(map(int, self.recommendArg.topK.split(",")))
        self.agentArg = agentArg

        self.targetItem = None
        self.targetItemRaw = None
        self.rawRecommendresult = None

        self.current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.logger = logging.getLogger(self.recommendModelName + "_SkillRec")
        self.logger.setLevel(logging.INFO)

        os.makedirs("./log/skillrec/", exist_ok=True)
        logFilename = f"{self.recommendModelName}_{self.datasetName}_{self.current_time}.log"
        handler = logging.FileHandler(f"./log/skillrec/{logFilename}")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(handler)

        message = "\n\n---------- Skill-Centric Recommender Information ----------\n"
        for k, v in vars(recommendArg).items():
            message += f"{k}: {v}\n"
        if agentArg:
            message += "\n---------- Agent Parameters ----------\n"
            for k, v in vars(agentArg).items():
                message += f"{k}: {v}\n"
        self.logger.info(message)
        print(message)

    def RecommendTrain(self):
        """
        New workflow:
          1) load/train tool model
          2) prepare prior scores and candidate-side artifacts
          3) initialize skill bank from train samples
          4) update skill bank on train samples
        """
        os.makedirs(self.recommendArg.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.recommendArg.save_dir, self.recommendModelName), exist_ok=True)

        # --------------------------------------------------
        # 1. load or train tool model (e.g. SASRec wrapper)
        # --------------------------------------------------
        if self.agent and self.tool_name is not None:
            print(f"Loading tool model {self.tool_name} for skill-centric recommendation...")
            module = importlib.import_module("recommender." + self.tool_type + "." + self.tool_name)
            ToolClass = getattr(module, self.tool_name)

            checkpoint_dir = os.path.join(self.recommendArg.save_dir, self.recommendModelName)
            save_path = os.path.join(
                checkpoint_dir,
                self.datasetName,
                "clean",
                "tool",
                f"{self.tool_name}_model.pth",
            )

            config_file_list = []
            if self.agentArg and self.agentArg.config_file_path:
                config_file_list = [self.agentArg.config_file_path]
            else:
                default_conf_path = os.path.join(
                    "tool_conf",
                    self.recommendModelName,
                    self.datasetName,
                    f"{self.tool_name}.yaml",
                )
                if os.path.exists(default_conf_path):
                    config_file_list = [default_conf_path]
                else:
                    config_file_list = [os.path.join("tool_conf", "default", f"{self.tool_name}.yaml")]
                    print(f"Warning: Using default config {config_file_list[0]}")

            dir_path = os.path.join(
                self.recommendArg.save_dir,
                self.recommendModelName,
                self.datasetName,
                "clean",
                "tool",
            )

            tool_instance = ToolClass(
                self.recommendArg,
                self.recommendModel.data,
                config_file_list,
                dir_path,
                self.agentArg,
            )

            if not os.path.exists(save_path):
                print(f"Training tool model to {save_path}...")
                tool_instance.train(save_path)
            else:
                print(f"Tool model found at {save_path}")

            if hasattr(self.recommendModel, "data"):
                tool_instance.arlib_dataset = self.recommendModel.data

            self.recommendModel.tool_model = tool_instance.load(save_path)

        # --------------------------------------------------
        # 2. prepare save paths
        # --------------------------------------------------
        save_dir = os.path.join(
            self.recommendArg.save_dir,
            self.recommendModelName,
            self.datasetName,
            "clean",
        )
        root_dir = os.path.join(
            self.recommendArg.save_dir,
            self.recommendModelName,
            self.datasetName,
        )

        # --------------------------------------------------
        # 3. load recommender if needed
        # --------------------------------------------------
        if self.agent and self.agentArg and self.agentArg.load_recommender:
            print(f"Loading existing recommender from {save_dir}")
            self.recommendModel.load(save_dir, root_dir)
        else:
            print("Training recommender model from scratch...")
            if getattr(self.recommendArg, "save", False):
                self.recommendModel.save(save_dir)

        # --------------------------------------------------
        # 4. generate prior scores / candidate artifacts
        # --------------------------------------------------
        self.recommendModel.train(save_dir, root_dir)

        # --------------------------------------------------
        # 5. train-pass 1: initialize skill bank
        # --------------------------------------------------
        if self.agentArg and getattr(self.agentArg, "enable_memskill", False):
            print("[ARLib] Bootstrap skill bank from train samples...")
            self.recommendModel.bootstrap_skill_bank_from_train(
                save_dir=save_dir,
                root_dir=root_dir,
            )

            # --------------------------------------------------
            # 6. train-pass 2: update skill bank on train samples
            # --------------------------------------------------
            print("[ARLib] Generate trajectories and refine the skill bank...")
            self.recommendModel.update_skill_bank_on_train(
                save_dir=save_dir,
                root_dir=root_dir,
            )

            print("[ARLib] Generate teacher trajectories for SFT...")
            self.recommendModel.generate_skill_distill_data(
                save_dir=save_dir,
                root_dir=root_dir,
            )

    def PrepareForTest(self):
        """Load frozen test artifacts without entering any training/build/evolution path."""
        if not self.agent or self.tool_name is None:
            raise RuntimeError("Test-only ARS requires an agent tool model.")

        module = importlib.import_module("recommender." + self.tool_type + "." + self.tool_name)
        ToolClass = getattr(module, self.tool_name)
        save_path = os.path.join(
            self.recommendArg.save_dir,
            self.recommendModelName,
            self.datasetName,
            "clean",
            "tool",
            f"{self.tool_name}_model.pth",
        )
        if not os.path.isfile(save_path):
            raise FileNotFoundError(
                f"Frozen tool checkpoint is required in --test_only mode: {save_path}"
            )

        if self.agentArg and self.agentArg.config_file_path:
            config_file_list = [self.agentArg.config_file_path]
        else:
            model_conf = os.path.join(
                "tool_conf", self.recommendModelName, self.datasetName, f"{self.tool_name}.yaml"
            )
            config_file_list = [
                model_conf if os.path.exists(model_conf)
                else os.path.join("tool_conf", "default", f"{self.tool_name}.yaml")
            ]

        tool_dir = os.path.dirname(save_path)
        tool_instance = ToolClass(
            self.recommendArg,
            self.recommendModel.data,
            config_file_list,
            tool_dir,
            self.agentArg,
        )
        tool_instance.arlib_dataset = self.recommendModel.data
        self.recommendModel.tool_model = tool_instance.load(save_path)

        root_dir = os.path.join(
            self.recommendArg.save_dir,
            self.recommendModelName,
            self.datasetName,
        )
        save_dir = os.path.join(root_dir, "clean")
        print(f"[ARLib] TEST-ONLY: loaded frozen tool checkpoint {save_path}")
        print("[ARLib] TEST-ONLY: no Skill Bank build/evolution, retriever training, or LoRA training will run")
        self.recommendModel.prepare_for_test(save_dir=save_dir, root_dir=root_dir)

    def RecommendTest(self):
        if self.model_type == "agent":
            save_dir = os.path.join(
                self.recommendArg.save_dir,
                self.recommendModelName,
                self.datasetName,
                "clean",
            )
            root_dir = os.path.join(
                self.recommendArg.save_dir,
                self.recommendModelName,
                self.datasetName,
            )

            self.rawRecommendresult = self.recommendModel.test(save_dir, root_dir)
            message = f"Skill-centric recommender {self.recommendModelName} tested on clean data\n"

            if hasattr(self, "targetItem") and self.targetItem is not None:
                agentmetrics = AgentMetric(self.recommendModel, self.targetItem, self.top)

                overall_precision = agentmetrics.precision()["overall"]
                overall_hr = agentmetrics.hitRate()["overall"]
                overall_recall = agentmetrics.recall()["overall"]
                overall_ndcg = agentmetrics.NDCG()["overall"]

                message += "\n---------- Overall Recommendation Performance ----------\n"
                for i, k in enumerate(self.top):
                    message += (
                        f"Top-{k}: Precision={overall_precision[i]:.4f}, "
                        f"HR={overall_hr[i]:.4f}, "
                        f"Recall={overall_recall[i]:.4f}, "
                        f"NDCG={overall_ndcg[i]:.4f}\n"
                    )

            self.logger.info(message)
            print(message)

        else:
            _, self.rawRecommendresult = self.recommendModel.test()
            message = f"Recommender {self.recommendModelName} tested on clean data\n"
            for line in self.rawRecommendresult:
                message += str(line) + "\n"
            self.logger.info(message)
            print(message)

    def set_target_item(self, target_item):
        self.targetItem = target_item
        if hasattr(self.recommendModel, "data") and hasattr(self.recommendModel.data, "id2item"):
            if isinstance(target_item, list):
                self.targetItemRaw = [
                    self.recommendModel.data.id2item.get(tid, tid) for tid in target_item
                ]
            else:
                self.targetItemRaw = self.recommendModel.data.id2item.get(target_item, target_item)
