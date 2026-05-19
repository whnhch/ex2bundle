from rouge_score import rouge_scorer

class EvaluationScore:
    def compareScore(self,paql_summary,ground_truth):
        scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        scores = scorer.score(paql_summary,ground_truth)
        #print(scores)
        val = []
        for key,values in scores.items():
            values = str(values)
            values = values.split(",")
            p = float(values[0][16:])
            r = float(values[1][8:])
            f1 = float(values[2][10:-1])
            if (p == 0 and r == 0):
                f2 = 0.0
            else:
                f2 = (5.0*p*r)/(4.0*(p+r))
            val.append([p,r,f1,f2])
        return val