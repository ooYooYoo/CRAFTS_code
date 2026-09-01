import math
import numpy as np

class RecommendMetric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return hit_num/total_num

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return prec / (len(hits) * N)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = sum(recall_list) / len(recall_list)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res.keys():
            for doubleEntry in res[entry]:
                error+=abs(doubleEntry[0]-doubleEntry[1])
                count+=1
        if count==0:
            return error
        return error/count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res.keys():
            for doubleEntry in res[entry]:
                error += (doubleEntry[0]-doubleEntry[1])**2
                count += 1
        if count == 0:
            return error
        return math.sqrt(error/count)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / len(res)

def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = RecommendMetric.hits(origin, predicted)
        hr = RecommendMetric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = RecommendMetric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = RecommendMetric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        #MAP = Measure.MAP(origin, predicted, n)
        #indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = RecommendMetric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure

def rating_evaluation(res):
    measure = []
    mae = RecommendMetric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = RecommendMetric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure


class AttackMetric(object):
    """
    param 
    targetItem:list, targetItem: id
    """
    def __init__(self, recommendModel, targetItem, top=[10]):
        self.recommendModel = recommendModel
        self.targetItem = targetItem
        self.top = top

    def precision(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += k
            for j in self.targetItem:
                for k in range(len(self.top)):
                    if j in result[k]:
                        hit[k] += 1
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def hitRate(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += 1
            for k in range(len(self.top)):
                hit[k] += int(len(set(self.targetItem) & set(result[k])) > 0)/len(self.targetItem)
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def recall(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += len(self.targetItem)
            for j in self.targetItem:
                for k in range(len(self.top)):
                    if j in result[k]:
                        hit[k] += 1
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def NDCG(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                idcg=0
                for s in range(k):
                    if s < len(self.targetItem):
                        idcg+= 1 / np.log2(2 + s)
                totalNum[n] += idcg
            for step,r in enumerate(result):
                for rank,j in enumerate(r):
                    if j in self.targetItem:
                        hit[step]+=1 / np.log2(2 + rank)
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

class DetectMetric(object):
    def __init__(self):
        pass
    
    @staticmethod
    def accuracy(predict, label):
        num = 0
        for i in predict.keys():
            if predict[i] == label[i]: num += 1
        return num / len(predict.keys())

    @staticmethod
    def recall(predict, label):
        num = 0
        Num = 0
        for i in predict.keys():
            if label[i] == 1: Num += 1
            if label[i] == 1 and predict[i] == 1: num += 1
        return num / Num

    @staticmethod
    def precision(predict, label):
        num = 0
        Num = 0
        for i in predict.keys():
            if predict[i] == 1 and label[i] == 1: num += 1
            if predict[i] == 1: Num += 1
        if Num == 0: return 0
        return num / Num   

    @staticmethod
    def f1(predict, label):
        pre = DetectMetric.precision(predict, label)
        recall = DetectMetric.recall(predict, label)
        if pre + recall == 0: return 0
        return (2 * pre * recall) / (pre + recall)

class RecommendMetricRatings(object):
    def __init__(self):
        self.rating_threshold = 4

    @staticmethod
    def hit(origin, res, rating_threshold=4):
        hit_count = {}
        for user in origin:
            
            items = {item: rating for item, rating in origin[user].items() if rating >= rating_threshold}
            predicted = set(item[0] for item in res[user])
            hit_count[user] = len(items.keys() & predicted)
            #print(hit_count)
        return hit_count

    @staticmethod
    def HR_K(origin, hits, rating_threshold=4):
        
        total_num = len(origin)
        if total_num == 0:
            return 0

        
        hit_num = sum(1 for user in hits if hits[user] >= 1)
        #print(total_num, hit_num)
        return hit_num / total_num

    @staticmethod
    def CG(res):
        total_gain = 0
        for user in res:
            gain = 0
            for rank, item in enumerate(res[user]):
                gain += 1.0 / math.log2(rank + 2)
            total_gain += gain
        return total_gain / len(res)

    @staticmethod
    def PS(origin, res):
        shift_sum = 0
        count = 0
        for user in res:
            for item in res[user]:
                if item[0] in origin[user]:
                    shift_sum += abs(origin[user][item[0]] - item[1])
                    count += 1
        return shift_sum / count if count > 0 else 0

    @staticmethod
    def F1_score(origin, res):
        precision = 0
        recall = 0
        for user in origin:
            true_positives = len(set(origin[user].keys()).intersection(set(item[0] for item in res[user])))
            predicted_positives = len(res[user])
            actual_positives = len(origin[user])

            if predicted_positives > 0:
                precision += true_positives / predicted_positives
            if actual_positives > 0:
                recall += true_positives / actual_positives

        precision /= len(origin)
        recall /= len(origin)

        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0

def ranking_evaluation2(origin, res, N):
    measure = []
    metric = RecommendMetric()  
    for n in N:
        predicted = {user: res[user][:n] for user in res}
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)

        hits = metric.hit(origin, predicted, metric.rating_threshold)
        hr_k = metric.HR_K(origin, hits)
        indicators.append('HR_K: {}'.format(hr_k) + '\n')

        cg = RecommendMetric.CG(predicted)
        indicators.append('CG: ' + str(cg) + '\n')

        ps = RecommendMetric.PS(origin, predicted)
        indicators.append('PS: ' + str(ps) + '\n')

        f1 = RecommendMetric.F1_score(origin, predicted)
        indicators.append('F1_Score: ' + str(f1) + '\n')

        measure.append('Top ' + str(n) + '\n')
        measure += indicators

    return measure

class AttackMetricRatings(object):
    def __init__(self, recommendModel, targetItem, top=[10]):
        self.recommendModel = recommendModel
        self.targetItem = targetItem
        self.top = top
        self.rating_threshold = 4  

    def HR_K(self):
        totalNum = [0] * len(self.top)  
        hit = [0] * len(self.top)  

        
        for user in self.recommendModel.data.user:
            score = self.recommendModel.predict(user)  
            result = []

            
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])  
                totalNum[n] += 1  

            
            for n in range(len(self.top)):
                
                valid_hits = [item for item in result[n] if
                              self.recommendModel.predict(user)[item] >= self.rating_threshold]
                
                hit[n] += int(len(set(self.targetItem) & set(valid_hits)) > 0)

        
        # print(hit,totalNum)
        return [hit[i] / totalNum[i] if totalNum[i] > 0 else 0 for i in range(len(self.top))]

    def precision(self):
        totalNum = [0] * len(self.top)
        hit = [0] * len(self.top)
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += k
            for j in self.targetItem:
                for k in range(len(self.top)):
                    if j in result[k]:
                        hit[k] += 1

        return [hit[i] / totalNum[i] for i in range(len(self.top))]

    def CG(self):
        total_gain = 0
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = np.argsort(-score)
            gain = 0
            for rank, item in enumerate(result):
                gain += 1.0 / np.log2(rank + 2)
            total_gain += gain
        return [total_gain / len(self.recommendModel.data.user)]

    def avgScore(self):
        score_sum = 0
        count = 0
        for i in self.recommendModel.data.user:
            predicted_scores = self.recommendModel.predict(i)
            for item in self.targetItem:
                score_sum += predicted_scores[item]  
                count += 1
        return [score_sum / count] if count > 0 else [0]

    def F1_score(self):
        precision = self.precision()
        hr_k = self.HR_K()
        f1_scores = []
        for i in range(len(self.top)):
            if precision[i] + hr_k[i] > 0:
                f1 = 2 * (precision[i] * hr_k[i]) / (precision[i] + hr_k[i])
            else:
                f1 = 0
            f1_scores.append(f1)
        return f1_scores


class AgentMetric(object):
    
    def __init__(self, recommendModel, targetItem, top=[10]):
        
        self.recommendModel = recommendModel
        self.targetItem = targetItem
        self.top = top
        self.num_items = len(self.recommendModel.data.item)
        
        
        self.test_ground_truth = {}
        if hasattr(self.recommendModel.data, 'test_data'):
            for entry in self.recommendModel.data.test_data:
                user_id = entry[0]  
                item_id = entry[1]  
                if user_id not in self.test_ground_truth:
                    self.test_ground_truth[user_id] = set()
                self.test_ground_truth[user_id].add(item_id)
        
    def _convert_agent_results_to_scores(self, agent_results):
        
        score_dict = {}
        
        for user_id, result in agent_results.items():
            
            scores = np.zeros(self.num_items)
            
            if isinstance(result, list):
                
                for rank, item_id in enumerate(result):
                    if 0 <= item_id < self.num_items:
                        scores[item_id] = 1e9 - rank
            
            elif isinstance(result, dict):
                
                for item_id, score in result.items():
                    if 0 <= item_id < self.num_items:
                        scores[item_id] = score
            
            score_dict[user_id] = scores
        
        return score_dict
    
    def precision(self):
        
        if not hasattr(self.recommendModel, 'agent_results'):
            raise ValueError("Model must have 'agent_results' attribute")
        
        score_dict = self._convert_agent_results_to_scores(self.recommendModel.agent_results)
        
        
        overall_total = [0 for _ in range(len(self.top))]
        overall_hit = [0 for _ in range(len(self.top))]
        
        
        target_total = [0 for _ in range(len(self.top))]
        target_hit = [0 for _ in range(len(self.top))]
        
        for user_id, scores in score_dict.items():
            for n, k in enumerate(self.top):
                topk_items = set(np.argsort(-scores)[:k])
                
                
                if user_id in self.test_ground_truth:
                    true_items = self.test_ground_truth[user_id]
                    overall_total[n] += k
                    overall_hit[n] += len(topk_items & true_items)
                
                
                target_total[n] += k
                target_hit[n] += len(topk_items & set(self.targetItem))
        
        overall_precision = [overall_hit[i] / overall_total[i] if overall_total[i] > 0 else 0 
                           for i in range(len(self.top))]
        
        target_precision = [target_hit[i] / target_total[i] if target_total[i] > 0 else 0 
                          for i in range(len(self.top))]
        
        return {
            'overall': overall_precision,
            'target': target_precision
        }
    
    def hitRate(self):
        
        score_dict = self._convert_agent_results_to_scores(self.recommendModel.agent_results)
        
        overall_total = [0 for _ in range(len(self.top))]
        overall_hit = [0 for _ in range(len(self.top))]
        
        target_total = [0 for _ in range(len(self.top))]
        target_hit = [0 for _ in range(len(self.top))]
        
        for user_id, scores in score_dict.items():
            for n, k in enumerate(self.top):
                topk_items = set(np.argsort(-scores)[:k])
                
                
                if user_id in self.test_ground_truth:
                    true_items = self.test_ground_truth[user_id]
                    overall_total[n] += 1
                    if len(topk_items & true_items) > 0:
                        overall_hit[n] += 1
                
                
                target_total[n] += 1
                if len(topk_items & set(self.targetItem)) > 0:
                    target_hit[n] += 1
        
        overall_hr = [overall_hit[i] / overall_total[i] if overall_total[i] > 0 else 0 
                     for i in range(len(self.top))]
        
        target_hr = [target_hit[i] / target_total[i] if target_total[i] > 0 else 0
                    for i in range(len(self.top))]
        
        return {
            'overall': overall_hr,
            'target': target_hr
        }
    
    def recall(self):
        
        score_dict = self._convert_agent_results_to_scores(self.recommendModel.agent_results)
        
        overall_total = [0 for _ in range(len(self.top))]
        overall_hit = [0 for _ in range(len(self.top))]
        
        target_total = [0 for _ in range(len(self.top))]
        target_hit = [0 for _ in range(len(self.top))]
        
        for user_id, scores in score_dict.items():
            for n, k in enumerate(self.top):
                topk_items = set(np.argsort(-scores)[:k])
                
                
                if user_id in self.test_ground_truth:
                    true_items = self.test_ground_truth[user_id]
                    overall_total[n] += len(true_items)
                    overall_hit[n] += len(topk_items & true_items)
                
                
                target_total[n] += len(self.targetItem)
                target_hit[n] += len(topk_items & set(self.targetItem))
        
        overall_recall = [overall_hit[i] / overall_total[i] if overall_total[i] > 0 else 0 
                         for i in range(len(self.top))]
        
        target_recall = [target_hit[i] / target_total[i] if target_total[i] > 0 else 0 
                        for i in range(len(self.top))]
        
        return {
            'overall': overall_recall,
            'target': target_recall
        }
    
    def NDCG(self):
        
        score_dict = self._convert_agent_results_to_scores(self.recommendModel.agent_results)
        
        overall_dcg = [0 for _ in range(len(self.top))]
        overall_idcg = [0 for _ in range(len(self.top))]
        
        target_dcg = [0 for _ in range(len(self.top))]
        target_idcg = [0 for _ in range(len(self.top))]
        
        for user_id, scores in score_dict.items():
            for n, k in enumerate(self.top):
                topk_items = np.argsort(-scores)[:k]
                
                
                if user_id in self.test_ground_truth:
                    true_items = self.test_ground_truth[user_id]
                    
                    
                    for rank, item_id in enumerate(topk_items):
                        if item_id in true_items:
                            overall_dcg[n] += 1 / np.log2(2 + rank)
                    
                    
                    for rank in range(min(k, len(true_items))):
                        overall_idcg[n] += 1 / np.log2(2 + rank)
                
                
                target_items = set(self.targetItem)
                
                
                for rank, item_id in enumerate(topk_items):
                    if item_id in target_items:
                        target_dcg[n] += 1 / np.log2(2 + rank)
                
                
                for rank in range(min(k, len(self.targetItem))):
                    target_idcg[n] += 1 / np.log2(2 + rank)
        
        overall_ndcg = [overall_dcg[i] / overall_idcg[i] if overall_idcg[i] > 0 else 0 
                       for i in range(len(self.top))]
        
        target_ndcg = [target_dcg[i] / target_idcg[i] if target_idcg[i] > 0 else 0 
                      for i in range(len(self.top))]
        
        return {
            'overall': overall_ndcg,
            'target': target_ndcg
        }
    
    def avgScore(self):
        
        score_dict = self._convert_agent_results_to_scores(self.recommendModel.agent_results)
        
        
        overall_score_sum = 0
        overall_count = 0
        
        for user_id, scores in score_dict.items():
            if user_id in self.test_ground_truth:
                true_items = self.test_ground_truth[user_id]
                for item_id in true_items:
                    if 0 <= item_id < len(scores):
                        overall_score_sum += scores[item_id]
                        overall_count += 1
        
        
        target_score_sum = 0
        target_count = 0
        
        for user_id, scores in score_dict.items():
            for target_item in self.targetItem:
                if 0 <= target_item < len(scores):
                    target_score_sum += scores[target_item]
                    target_count += 1
        
        overall_avg = overall_score_sum / overall_count if overall_count > 0 else 0
        target_avg = target_score_sum / target_count if target_count > 0 else 0
        
        return {
            'overall': [overall_avg],
            'target': [target_avg]
        }
    
    def compute_all_metrics(self):
        
        return {
            'precision': self.precision(),
            'hitRate': self.hitRate(),
            'recall': self.recall(),
            'NDCG': self.NDCG(),
            'avgScore': self.avgScore()
        }
