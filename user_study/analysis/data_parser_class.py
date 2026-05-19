from utils.EvaluationScore import EvaluationScore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
np.random.seed(0)
from scipy.spatial.distance import cosine, jensenshannon
from collections import Counter
from datetime import datetime
from dateutil import parser



class DataParser():

    def __init__(self, data_path, sudocu_data_path, shared_docs_path, n_topics=10) -> None:
        self.data_path_ = data_path
        self.user_interaction_data_ = {}
        self.sudocu_datapath_ = sudocu_data_path
        self.shared_docs_path_ = shared_docs_path
        self.nTopics_ = n_topics
        self._load_sudocu_dataset()
        self.topic_names = np.array(['topic_' + str(i) for i in range(self.nTopics_)])
        with open(os.path.join(self.shared_docs_path_, "state_indices.txt")) as f:
            file = f.read()
            f.close()
        self.doc_indices = json.loads(file)

        self.eval_scorer = EvaluationScore()

        self.processed_user_interactions = {"sudocu":{}, "sbert":{}}

        self.upper_bounds = {"example":[], "relax":[]}
        self.lower_bounds = {"example":[], "relax":[]}
        self.all_bounds = {"example":[], "relax":[]}
        self.sum_bounds = {"example":[], "relax":[]}
        self.rouge_scores = []
        self.slider_topic_dists = []
        self.user_interaction_times = {}

    def process_dataset(self):
        self.process_user_sbert_data()
        self.process_user_sudocu_data()


    def _load_sudocu_dataset(self):
        self.df = pd.read_csv(self.sudocu_datapath_)
        self.sentences = self.df["sentence"]


    def extract_user_data(self, user_ids):
        raw_data = None

        with open(self.data_path_, "r") as f:
            raw_data = json.load(f)

        # print(raw_data.keys())

        for u_id in user_ids:
            data_interaction = {}
            data_interaction["sudocu"] = json.loads(raw_data[u_id]["SuDoCuTask"]["data_interactions"]["answer"])
            data_interaction["sbert"] = json.loads(raw_data[u_id]["SbertTask"]["data_interactions"]["answer"])

            # print(raw_data[u_id]["SuDoCuTask"]["data_interactions"]["answer"])
            # print(raw_data[u_id]["SbertTask"]["data_interactions"])

            start = json.loads(raw_data[u_id]["SuDoCuTask"]["data_interactions"]["answer"])[0].split(",")[2]
            end = json.loads(raw_data[u_id]["SuDoCuTask"]["data_interactions"]["answer"])[-1].split(",")[2]
            print(start)
            print(end)

            start_t = parser.parse(start[:-4])
            end_t = parser.parse(end[:-4])
            # date_object = datetime.strptime(start[:-4], '%d %b %Y %H:%M:%S').date()
            # print(type(date_object))
            # print(date_object)  # printed in default format
            diff = end_t - start_t
            sudocu_minutes = diff.total_seconds() / 60
            print('Total difference in minutes (sudocu): ', sudocu_minutes)
            
            start = json.loads(raw_data[u_id]["SbertTask"]["data_interactions"]["answer"])[0].split(",")[2]
            end = json.loads(raw_data[u_id]["SbertTask"]["data_interactions"]["answer"])[-1].split(",")[2]
            print(start)
            print(end)

            start_t = parser.parse(start[:-4])
            end_t = parser.parse(end[:-4])
            # date_object = datetime.strptime(start[:-4], '%d %b %Y %H:%M:%S').date()
            # print(type(date_object))
            # print(date_object)  # printed in default format
            diff = end_t - start_t
            sbert_minutes = diff.total_seconds() / 60
            print('Total difference in minutes (sbert): ', sbert_minutes)

            self.user_interaction_times[u_id] = {"sudocu":sudocu_minutes, "sbert":sbert_minutes}
            # time_diff = 

            self.user_interaction_data_[u_id] = data_interaction


    def process_user_sbert_data(self):

        for user_id, raw_user_data in self.user_interaction_data_.items():
            temp_user_data = {"example_summaries":{"used_keys":set(), "summary_update_counts":{}},
                              "learned_preferences":{"learned_count":-1, "preference_stats":[]},
                              "generated_summaries":{"count":0, "summary_stats":[]}}
            raw_user_sbert_data = raw_user_data["sbert"]

            print("User ID: {:s} had {:d} SBERT interactions with".format(user_id, len(raw_user_sbert_data)))

            for entry in raw_user_sbert_data:
                entry_data = entry.split(",")

                if entry_data[3] == "json_example" or entry_data[3] == "json_example_update":
                    self._extract_example_summary_data(temp_user_data, entry_data)

                elif entry_data[3] == "json_learn_summary":
                   self._extract_learned_preferences_data(temp_user_data, entry_data)
                #    print(exam_entry_data)

                elif entry_data[3] == "json_generate_summary":
                    self._extract_generated_summary_data_sbert(temp_user_data, entry_data)
            
            self.processed_user_interactions["sbert"][user_id] = temp_user_data


    def extract_meta_stats_sbert(self, user_ids, output_path=""):
        
        
        #  lists to hold data across users
        temp_user_list_raw = []
        temp_user_list_avgs = []

        # iterate over the users
        for u_id in user_ids:
            print(u_id)

            # extract this users data
            u_data = self.processed_user_interactions["sbert"][u_id]

            # array to aggregate all the data
            state_ids = []
            sbert_scores = []
            rouge_scores = []
            pred_topic_dists = []
            topic_bounds_initial = []
            pref_set_pointers = []

            # Rouge-1, Rouge-2, Rouge-L: [p, r, f1, f2]
            # First, we want to pull out all the rouge scores AND find avg +\- stddev
            for pred_summary in u_data["generated_summaries"]["summary_stats"]:
                # print(np.array(pred_summary["rouge_scores"]).shape)

                # store the type of relaxation this method employed
                sbert_scores.append(pred_summary["sim_scores"])
                # print(len(pred_summary["sim_scores"]))
                rouge_scores.append(pred_summary["rouge_scores"])
                # print(np.array(pred_summary["rouge_scores"]).shape)
                pred_topic_dists.append(pred_summary["topic_dists"])
                # print(np.array(pred_summary["topic_dists"]).shape)
                topic_bounds_initial.append(pred_summary["bounds"])
                pref_set_pointers.append(pred_summary["per_set_pointer"])
                state_ids.append(pred_summary["state_id"])


            # print(np.array(rouge_scores).shape)
            
                        
            # we also need some stats from the example summaries
            example_topic_bounds = []
            example_topic_avg_dists = []

            for pref_set_idx in pref_set_pointers:
                example_topic_bounds.append(u_data["learned_preferences"]["preference_stats"][pref_set_idx]["topic_bounds"])
                example_topic_avg_dists.append(np.squeeze(np.mean(u_data["learned_preferences"]["preference_stats"][pref_set_idx]["topic_scores"], axis=0)).tolist())

            

            # now... we want to populate the per-sample df entries
            for i in range(len(u_data["generated_summaries"]["summary_stats"])):
                
                # iterate over all the examples for each user
                    temp_user_entry = []
                    
                    # add the user_id
                    temp_user_entry.append(u_id)

                    # add the predicted summaries state id
                    temp_user_entry.append(state_ids[i])

                    # print(np.array(rouge_scores[i][j]).shape)

                    # "unwind" the components of the rouge-scores
                    # we want r1[p, r, f1], r2[p, r, f1], and r3[p, r, f1]
                    for _scores in np.mean(rouge_scores[i], axis=0):
                        for _sub_score in _scores:
                            temp_user_entry.append(_sub_score)

                    # add the sbert-similarity score
                    temp_user_entry.append(np.mean(sbert_scores[i], axis=0))

                    # print(sbert_scores[i][j])
                    # print(sbert_scores[i])

                    # add the bound difference between pred and avg examples
                    bound_diff = (np.array(example_topic_bounds[i]) - np.array(topic_bounds_initial[i])).tolist()
                    # print(bound_diff)
                    # print(bound_diff)
                    temp_list = []
                    for _b_diff in bound_diff:
                        temp_list.extend(_b_diff)

                    temp_user_entry.extend(temp_list)

                                        # calculate the abs(upper + lower bounds) stas v. the examples
                    sum_ex_bounds = np.sum(np.absolute(example_topic_bounds[i]), axis=1)

                    sum_pred_bounds_initial = np.sum(np.absolute(topic_bounds_initial[i]), axis=1)

                    sum_bound_diffs_ex = (sum_ex_bounds - sum_pred_bounds_initial).tolist()

                    temp_user_entry.extend(sum_bound_diffs_ex)

                    # print(len(temp_list))

                    # print(topic_bounds_initial[i])
                    # print(topic_bounds_relaxed[i])
                    # print(temp_list)

                    # print(temp_list)

                    # add the topic-dist difference
                    # print(example_topic_avg_dists[i])
                    # print(np.mean(pred_topic_dists[i], axis=0))
                    topic_diffs = (np.array(example_topic_avg_dists[i]) - np.mean(pred_topic_dists[i], axis=0)).tolist()
                    temp_user_entry.extend(topic_diffs)


                    # print(len(topic_diffs))

                    temp_user_list_raw.append(temp_user_entry)

                    # print(len(temp_user_entry))
            
            
        # end loop over users
        rouge_types = ["r1", "r2", "r3"]
        rouge_subtypes = ["p", "r", "f1", "f2"]

        bound_diff_types = ["example"]
        bound_lims = ["up-", "low-"]

        raw_column_names = ["user_id", "state_id"]

        # add the rouge-score column names
        for r_type in rouge_types:
            for r_sub_type in rouge_subtypes:
                raw_column_names.append(r_type + r_sub_type)

        # add sbert cosine-sim labels
        raw_column_names.append("cosine")

        # add the bound names
        for b_type in bound_diff_types:
            for t_name in self.topic_names:
                for _lim in bound_lims:
                    raw_column_names.append(_lim + b_type + "-" + t_name)

            for t_name in self.topic_names:
                raw_column_names.append(b_type + "-" + t_name)

        # add topic column names
        for t_name in self.topic_names:
            raw_column_names.append(t_name)

        # print(raw_column_names)
        # print(avg_column_names)

        # print(np.array(temp_user_list_raw).shape)

        self.sbert_example_df = pd.DataFrame(data=temp_user_list_raw, columns=raw_column_names)


    def process_user_sudocu_data(self):

        for user_id, raw_user_data in self.user_interaction_data_.items():
            temp_user_data = {"example_summaries":{"used_keys":set(), "summary_update_counts":{}},
                              "learned_preferences":{"learned_count":-1, "preference_stats":[]},
                              "generated_summaries":{"count":0, "summary_stats":[]}}
            raw_user_sudocu_data = raw_user_data["sudocu"]

            print("User ID: {:s} had {:d} SuDocu interactions!".format(user_id, len(raw_user_sudocu_data)))

            for entry in raw_user_sudocu_data:
                entry_data = entry.split(",")

                # print(entry_data[3])

                if entry_data[3] == "json_example" or entry_data[3] == "json_example_update":
                    self._extract_example_summary_data(temp_user_data, entry_data)

                elif entry_data[3] == "json_learn_summary":
                   self._extract_learned_preferences_data(temp_user_data, entry_data)
                #    print(exam_entry_data)

                elif entry_data[3] == "json_generate_summary" or entry_data[3] == "json_generate_summary_initial":
                    self._extract_generated_summary_data_sudocu(temp_user_data, entry_data)

            print("User ID: {:s} processed {:d} interactions!".format(user_id, len(temp_user_data)))
            
            self.processed_user_interactions["sudocu"][user_id] = temp_user_data


    def extract_meta_stats_sudocu(self, user_ids, output_path=""):
        #  lists to hold data across users
        temp_user_list_raw = []

        # iterate over the users
        for u_id in user_ids:
            print(u_id)

            # extract this users data
            u_data = self.processed_user_interactions["sudocu"][u_id]

            # array to aggregate all the data
            state_ids = []
            sbert_scores = []
            rouge_scores = []
            relax_types = []
            pred_topic_dists = []
            topic_bounds_initial = []
            topic_bounds_relaxed = []
            pref_set_pointers = []
            sentence_ids = []
            slider_values = []
            example_lengths = []

            # Rouge-1, Rouge-2, Rouge-L: [p, r, f1, f2]
            # First, we want to pull out all the rouge scores AND find avg +\- stddev
            for pred_summary in u_data["generated_summaries"]["summary_stats"]:
                # print(np.array(pred_summary["rouge_scores"]).shape)

                # store the type of relaxation this method employed
                relax_types.append("Doc" if pred_summary["modified_bounds"] == 0 else "Slider")
                sbert_scores.append(pred_summary["sim_scores"])
                rouge_scores.append(pred_summary["rouge_scores"])
                pred_topic_dists.append(pred_summary["topic_dists"])
                topic_bounds_initial.append(pred_summary["bounds"])
                topic_bounds_relaxed.append(pred_summary["relaxed_bounds"])
                pref_set_pointers.append(pred_summary["per_set_pointer"])
                state_ids.append(pred_summary["state_id"])
                sentence_ids.append(pred_summary["summary_indicies"])
                slider_values.append(pred_summary["slider_values"])
                example_lengths.append(u_data["learned_preferences"]["preference_stats"][pred_summary["per_set_pointer"]]["avg_summ_len"])

                print(pred_summary["state_id"])
                print(relax_types[-1])
                print(pred_summary["summary_indicies"])
                print(pred_summary["slider_values"])

                if len(slider_values) > 1:
                    print("prev-sliders: ", slider_values[-2])

                print("!!!!!!!!!!!!!!!!!!--------------------------!!!!!!!!!!!!!!!!!!!!\n")


                # print(len(pred_summary["sim_scores"]))
                # print(np.array(pred_summary["rouge_scores"]).shape)
                # print(np.array(pred_summary["topic_dists"]).shape)


            # print("Building user data entry")
            # # print(np.array(rouge_scores).shape)
            # print(len(sbert_scores[0]))
            # print(len(pred_topic_dists[0]))
            
            # print(sbert_scores[0])
            # print(pred_topic_dists[0])
            # we also need some stats from the example summaries
            example_topic_bounds = []
            example_topic_avg_dists = []

            for pref_set_idx in pref_set_pointers:
                example_topic_bounds.append(u_data["learned_preferences"]["preference_stats"][pref_set_idx]["topic_bounds"])
                example_topic_avg_dists.append(np.squeeze(np.mean(u_data["learned_preferences"]["preference_stats"][pref_set_idx]["topic_scores"], axis=0)).tolist())
                # print(example_topic_avg_dists[-1])

            # user
            user_slider_changes = {}
            prev_slider_doc_change = False
            
            # now... we want to populate the per-sample df entries
            for i in range(len(u_data["generated_summaries"]["summary_stats"])):
                
                example_has_diff_sliders = False

                # 
                print(np.sum((np.array(slider_values[i]) - np.array(slider_values[i-1]))))
                if relax_types[i] == "Slider" and state_ids[i] == state_ids[i-1] and (np.sum(np.array(slider_values[i]) - np.array(slider_values[i-1])) != 0 or prev_slider_doc_change):
                    print("Check Slider values for state-iD: ", state_ids[i])

                    if prev_slider_doc_change:
                        prev_slider_doc_change = False
                    
                    if state_ids[i] in user_slider_changes.keys():
                        store_items = True
                        # loop over checking if we already stored this diff already
                        for _prev, _new in zip(user_slider_changes[state_ids[i]]["prev"], user_slider_changes[state_ids[i]]["new"]):

                            if (Counter(slider_values[i-1]) == Counter(_prev) and Counter(slider_values[i]) == Counter(_new)):
                                store_items = False

                        if store_items:
                            print("prev-slider-values: ", slider_values[i-1])
                            print("current-slider-values: ", slider_values[i])
                            user_slider_changes[state_ids[i]]["prev"].append(slider_values[i-1])
                            user_slider_changes[state_ids[i]]["new"].append(slider_values[i])

                            print("!!!!Example has unique slider values!!!")

                            example_has_diff_sliders = True

                    else:
                        user_slider_changes[state_ids[i]] = {"prev":[], "new":[]}
                        print("prev-slider-values: ", slider_values[i-1])
                        print("current-slider-values: ", slider_values[i])
                        user_slider_changes[state_ids[i]]["prev"].append(slider_values[i-1])
                        user_slider_changes[state_ids[i]]["new"].append(slider_values[i])
                        example_has_diff_sliders = True
                
                elif relax_types[i] == "Slider" and state_ids[i] != state_ids[i-1] and np.sum((np.array(slider_values[i]) - np.array(slider_values[i-1]))) == 0:
                    prev_slider_doc_change = True

                temp_user_entry = []
                
                # add the user_id
                temp_user_entry.append(u_id)
                print("User-ID: ", u_id)

                # add the predicted summaries state id
                temp_user_entry.append(state_ids[i])
                print("State-ID: ", state_ids[i])

                # add the relaxation type
                temp_user_entry.append(relax_types[i])
                print("Relaxation Type: ", relax_types[i])

                # add preference set number
                temp_user_entry.append(pref_set_pointers[i])
                print("learned0pref. pointer: ", pref_set_pointers[i])

                # Denote if the sliders where different this time
                temp_user_entry.append(example_has_diff_sliders)
                print("If there was a slider modification: ", example_has_diff_sliders)

                # avg example summaries length
                temp_user_entry.append(example_lengths[i])
                print("Ex.-Summ-Len: ", example_lengths[i])

                print("Adding Rouge-Scores")
                # "unwind" the components of the rouge-scores
                # we want r1[p, r, f1], r2[p, r, f1], and r3[p, r, f1]
                for _scores in  np.mean(rouge_scores[i], axis=0):
                    for _sub_score in _scores:
                        temp_user_entry.append(_sub_score)
                        print("\t: ", _sub_score)

                # add the sbert-similarity score
                temp_user_entry.append(np.mean(sbert_scores[i]))
                print("Adding SBERT Score: ", np.mean(sbert_scores[i]))


                # print(sbert_scores[i])

                # print(example_topic_bounds[i])

                # Look into this!!!!!
                # add the bound difference between pred and avg examples
                bound_diff = (np.array(example_topic_bounds[i]) - np.array(topic_bounds_relaxed[i])).tolist()
                print("Adding the example-bound difference: ", bound_diff)
                temp_list = []
                for _b_diff in bound_diff:
                    temp_list.extend(_b_diff)

                temp_user_entry.extend(temp_list)

                # calculate the abs(upper + lower bounds) stas v. the examples
                sum_ex_bounds = np.sum(np.absolute(example_topic_bounds[i]), axis=1)

                sum_pred_bounds_relaxed = np.sum(np.absolute(topic_bounds_relaxed[i]), axis=1)

                sum_bound_diffs_ex = (sum_ex_bounds - sum_pred_bounds_relaxed).tolist()

                temp_user_entry.extend(sum_bound_diffs_ex)
                print("Adding the \"sum\"-based bound diffs (example): ", sum_bound_diffs_ex)


                # print(len(temp_list))

                # print(topic_bounds_initial[i])
                # print(topic_bounds_relaxed[i])
                # print(temp_list)

                # add the bound difference between submitted and relaxed bounds
                bound_diff = (np.array(topic_bounds_initial[i]) - np.array(topic_bounds_relaxed[i])).tolist()
                temp_list = []
                print("Adding the example-bound difference: ", bound_diff)
                for _b_diff in bound_diff:
                    temp_list.extend(_b_diff)

                # print(len(temp_list))
                
                temp_user_entry.extend(temp_list)

                # calculate the abs(upper + lower bounds) stas v. initial v relaxed
                sum_pred_bounds_initial = np.sum(np.absolute(topic_bounds_initial[i]), axis=1)
                # print(sum_pred_bounds_initial.shape)

                sum_bound_diffs_pred = (sum_pred_bounds_initial - sum_pred_bounds_relaxed).tolist()

                temp_user_entry.extend(sum_bound_diffs_pred)

                print("Adding the \"sum\"-based bound diffs (pred): ", sum_bound_diffs_pred)

                # # get the difference between the initial and 
                # if example_has_diff_sliders:
                #     # intial-bounds - prev relaxed bounds


                #     # relaxed-bounds - prev-relaxed-bounds



                # print(temp_list)

                # add the topic-dist difference
                # print(pred_topic_dists[i][j])
                # print(example_topic_avg_dists[i])
                # print("Pred-topic-dist avg shape: ", np.mean(pred_topic_dists[i], axis=0).shape)
                topic_diffs = (np.array(example_topic_avg_dists[i]) - np.mean(pred_topic_dists[i], axis=0)).tolist()
                print("Adding the example-topic-score difference: ", topic_diffs)
                temp_user_entry.extend(topic_diffs)

                # print(len(topic_diffs))

                # TODO: triple check all the stuff that leads to this:
                #     the slider values
                #     the sentence ids!
                if example_has_diff_sliders:
                    # find the last summary of the 
                    topic_diffs_slider = (np.array(np.mean(pred_topic_dists[i], axis=0)) - np.array(np.mean(pred_topic_dists[i-1], axis=0))).tolist()
                    temp_user_entry.extend(topic_diffs_slider)
                    # print(sentence_ids[i])
                    # print(sentence_ids[i-1])


                    # calculate number of added sentences
                    num_added = len(set(sentence_ids[i]).difference(sentence_ids[i-1]))
                    num_removed = len(set(sentence_ids[i-1]).difference(sentence_ids[i]))
                    total_change = num_added + num_removed
                    temp_user_entry.append(num_added)
                    temp_user_entry.append(num_removed)
                    temp_user_entry.append(total_change)
                    print("Num added: ", num_added)
                    print("Num removed: ", num_removed)
                    print("Total-change: ", total_change)



                temp_user_list_raw.append(temp_user_entry)

                print("-----------------------------------------------------------\n")

                    # print(len(temp_user_entry))
            
        # end loop over users
        rouge_types = ["r1", "r2", "r3"]
        rouge_subtypes = ["p", "r", "f1", "f2"]

        bound_diff_types = ["example", "relax"]
        bound_lims = ["up-", "low-"]

        raw_column_names = ["user_id", "state_id", "relax_type", "perf_pointer", "slider_mod", "evg_ex_len"]


        # add the rouge-score column names
        for r_type in rouge_types:
            for r_sub_type in rouge_subtypes:
                raw_column_names.append(r_type + r_sub_type)
                self.rouge_scores.append(r_type + r_sub_type)

        # add sbert cosine-sim labels
        raw_column_names.append("cosine")

        # add the bound names
        for b_type in bound_diff_types:
            for t_name in self.topic_names:
                for _lim in bound_lims:
                    raw_column_names.append(_lim + b_type + "-" + t_name)
                    
                    if _lim == bound_lims[0]:
                        self.upper_bounds[b_type].append(_lim + b_type + "-" + t_name)
                    else:
                        self.lower_bounds[b_type].append(_lim + b_type + "-" + t_name)

                    self.all_bounds[b_type].append(_lim + b_type + "-" + t_name)

            for t_name in self.topic_names:
                raw_column_names.append(b_type + "-" + t_name)
                self.sum_bounds[b_type].append(b_type + "-" + t_name)


        # add topic column names
        for t_name in self.topic_names:
            raw_column_names.append(t_name)

        for t_name in self.topic_names:
            raw_column_names.append("slider-" + t_name)
            self.slider_topic_dists.append("slider-" + t_name)


        raw_column_names.append("num_added")
        raw_column_names.append("num_removed")
        raw_column_names.append("total_change")

        print(raw_column_names)

        # print(np.array(temp_user_list_avgs).shape)
        
        print("Length of the user-data: ", len(temp_user_list_raw[0]))
        print(temp_user_list_raw[0])
        self.sudocu_sample_df = pd.DataFrame(data=temp_user_list_raw, columns=raw_column_names)

    
    # dict_keys(['modified_bounds', 'state_id', 'state_name', 'slider_values', 'bounds', 'relaxed_bounds', 'examples', 'summary_indicies', 'default_summ', 'model_feedback'])
    def _extract_generated_summary_data_sbert(self, _temp_user_data, data_entry):
        pref_entry_data = json.loads(",".join(data_entry[4:]))
        target_doc = pref_entry_data["state_name"]

        # remove nonsense we don't need
        pref_entry_data.pop('examples')

        # add a thing we want to ensure we track
        curr_pref_set = _temp_user_data["learned_preferences"]["learned_count"]
        pref_entry_data["per_set_pointer"] = curr_pref_set

        # pull out stuff we want
        predicted_summary = " ".join(self.sentences[pref_entry_data["summary_indicies"]])

        topic_dists = self.df[self.df['sid'].isin(pref_entry_data["summary_indicies"])][self.topic_names].to_numpy().tolist()

        # print(topic_dists)

        # calculate the bounds of the returned pred summary
        topic_sums = [np.sum(topic_dists, axis=0)]

        # print(topic_sums)

        # get the min and max of every topic from across all the summariess
        topic_min = list(np.min(topic_sums, axis=0))
        topic_max = list(np.max(topic_sums, axis=0))
        # avg_sum_len = avg_sum_len / len(example_summaires)

        # instantiate a numpy array to hold the generated topic bounds
        topic_query_bounds = np.zeros((self.nTopics_, 2))
        
        # I am not entirely sure why this and the rounding is here
        #     but this is the Vanillia SuDocu code
        eps = 1e-4
        
        for i in range(len(topic_min)):
            topic_query_bounds[i, 0] = round(topic_min[i] * 0.9 - eps, 4)
            topic_query_bounds[i, 1] = round(topic_max[i] * 1.1 + eps, 4)

        assert topic_query_bounds.shape[0] == self.nTopics_, "query bounds are not nTopics Long: SuDocuBase -> get_bounds"


        # print(np.array(topic_dists).shape)

        # load the NPZ file storing the SBERT embeddings for this document
        file_path = os.path.join(self.shared_docs_path_, "StateDocuments/", target_doc.strip() + "sudocu.npz")
        data = np.load(file_path)
        doc_sbert_embedding_data = data['embedding']

        # document index offset
        test_doc_idx_offset = self.doc_indices[target_doc][0]
        pred_sbert_embedding = np.mean(doc_sbert_embedding_data[[sum_idx - test_doc_idx_offset for sum_idx in pref_entry_data["summary_indicies"]]], axis=0)
        
        scores = []
        sim_scores = []
        all_embeddings = []


        for state_id in _temp_user_data["learned_preferences"]["preference_stats"][curr_pref_set]["state_ids"]:
            _data = _temp_user_data["example_summaries"][state_id]
            gt_summary = " ".join(_data["sentences"])
            avg_sbert_embedding = np.mean(_data["sbert_embeddings"], axis=0)
            all_embeddings.extend(_data["sbert_embeddings"])
            
            sim_scores.append(1.0 - cosine(pred_sbert_embedding, avg_sbert_embedding))

            r_sorces = self.eval_scorer.compareScore(predicted_summary, gt_summary)
            # print(predicted_summary)
            # print()
            # print(gt_summary)
            # print()
            # print(r_sorces)

            # print("\n---------------------------------------\n")
            

            scores.append(r_sorces)

        all_embeddings_avg = np.mean(all_embeddings, axis=0)
        avg_sim = 1.0 - cosine(pred_sbert_embedding, all_embeddings_avg)


        pref_entry_data["pred_sbert_embedding"] = pred_sbert_embedding
        pref_entry_data["rouge_scores"] = scores 
        pref_entry_data["sim_scores"] = sim_scores
        pref_entry_data["all_embeddings_avg"] = all_embeddings_avg
        pref_entry_data["avg_sim"] = avg_sim
        pref_entry_data["topic_dists"] = topic_dists
        pref_entry_data["bounds"] = topic_query_bounds.tolist()


        _temp_user_data["generated_summaries"]["summary_stats"].append(pref_entry_data)
        # scores = self.eval_scorer.compareScore(predicted_summary, gt_summary)
    

    def _process_model_feedback_sudocu(self, model_feedback):
        _lines = model_feedback.split("\n")

        usr_dat_dict = {"cplex_runtimes":[], "build_time":[], "load_time":[], "relax_time":[]}

        for line in _lines:

            if "Using Relaxation" in line:
                # print("Algorithm: ", line.split("#")[-1])
                usr_dat_dict["alg"] = line.split("#")[-1]
            
            if "Number of candidate sentences" in line:
                val = int(line.split(":")[-1].strip())
                # print("Num of sents: ", val)
                usr_dat_dict["problem_size"] = val

            if "CPLEX run-time:" in line:
                val = float(line.split(":")[-1].strip())
                # print("CPLEX-Time: ", val)
                usr_dat_dict["cplex_runtimes"].append(val)

            if "Solved ILP " in line:
                val = int(line.split("Solved ILP with ")[-1][0:-11].strip())
                # print("ILP-Iterations: ", val)
                usr_dat_dict["solve_iter"] = val

            if " Model build-time" in line:
                val = float(line.split("Model build-time: : ")[-1].strip())
                # print(val)
                usr_dat_dict["build_time"].append(val)

            if " - Data processing-time: " in line:
                val = float(line.split(" - Data processing-time: ")[-1].strip())
                # print(val)
                # print(val)
                usr_dat_dict["load_time"].append(val)

            if " - CPLEX refine-time:" in line:
                val = float(line.split(" - CPLEX refine-time: ")[-1].strip())
                # print(val)
                # print(val)
                usr_dat_dict["relax_time"].append(val)

        return usr_dat_dict


    # dict_keys(['modified_bounds', 'state_id', 'state_name', 'slider_values', 'bounds', 'relaxed_bounds', 'examples', 'summary_indicies', 'default_summ', 'model_feedback'])
    def _extract_generated_summary_data_sudocu(self, _temp_user_data, data_entry):
        pref_entry_data = json.loads(",".join(data_entry[4:]))
        target_doc = pref_entry_data["state_name"]

        # remove nonsense we don't need
        pref_entry_data.pop('examples')
        model_feedback = pref_entry_data.pop('model_feedback')

        summary_execution_dict = self._process_model_feedback_sudocu(model_feedback)

        # add a thing we want to ensure we track
        curr_pref_set = _temp_user_data["learned_preferences"]["learned_count"]
        pref_entry_data["per_set_pointer"] = curr_pref_set

        # pull out stuff we want
        predicted_summary = " ".join(self.sentences[pref_entry_data["summary_indicies"]])

        topic_dists = self.df[self.df['sid'].isin(pref_entry_data["summary_indicies"])][self.topic_names].to_numpy().tolist()

        # print(np.array(topic_dists).shape)

        # load the NPZ file storing the SBERT embeddings for this document
        file_path = os.path.join(self.shared_docs_path_, "StateDocuments/", target_doc.strip() + "sudocu.npz")
        data = np.load(file_path)
        doc_sbert_embedding_data = data['embedding']

        # document index offset
        test_doc_idx_offset = self.doc_indices[target_doc][0]
        pred_sbert_embedding = np.mean(doc_sbert_embedding_data[[sum_idx - test_doc_idx_offset for sum_idx in pref_entry_data["summary_indicies"]]], axis=0)
        
        scores = []
        sim_scores = []
        all_embeddings = []


        for state_id in _temp_user_data["learned_preferences"]["preference_stats"][curr_pref_set]["state_ids"]:
            # print(state_id)
            _data = _temp_user_data["example_summaries"][state_id]
            gt_summary = " ".join(_data["sentences"])
            avg_sbert_embedding = np.mean(_data["sbert_embeddings"], axis=0)
            all_embeddings.extend(_data["sbert_embeddings"])
            
            sim_scores.append(1.0 - cosine(pred_sbert_embedding, avg_sbert_embedding))

            r_sorces = self.eval_scorer.compareScore(predicted_summary, gt_summary)
            # print(predicted_summary)
            # print()
            # print(gt_summary)
            # print()
            # print(r_sorces)

            # print("\n---------------------------------------\n")
            

            scores.append(r_sorces)

        all_embeddings_avg = np.mean(all_embeddings, axis=0)
        avg_sim = 1.0 - cosine(pred_sbert_embedding, all_embeddings_avg)


        pref_entry_data["pred_sbert_embedding"] = pred_sbert_embedding
        pref_entry_data["rouge_scores"] = scores 
        pref_entry_data["sim_scores"] = sim_scores
        pref_entry_data["all_embeddings_avg"] = all_embeddings_avg
        pref_entry_data["avg_sim"] = avg_sim
        pref_entry_data["topic_dists"] = topic_dists
        pref_entry_data["model_feedback"] = summary_execution_dict


        _temp_user_data["generated_summaries"]["summary_stats"].append(pref_entry_data)
        # scores = self.eval_scorer.compareScore(predicted_summary, gt_summary)
                        

    def _extract_learned_preferences_data(self, _temp_user_data, data_entry):
        pref_data = {}

        # increment the learned-pref. counter
        _temp_user_data["learned_preferences"]["learned_count"] += 1
        
        # extract entry data
        pref_entry_data = json.loads(",".join(data_entry[4:]))
        state_ids = [_dict_["state_id"] for _dict_ in pref_entry_data]

        # now we want to find the avg +/- stddev of the topics scores / SBERT embeddings
        topic_scores = []
        sbert_embeddings = []

        mod_state_ids = {}

        summary_lengths = []

        for state_id in state_ids:
            # check for change counter and pull most recent change
            #   I.E. update state_id
            _update_count = _temp_user_data["example_summaries"]["summary_update_counts"][state_id]
            if _update_count > 0:
                _update_count -= 1
                mod_state_id = str(state_id) + "_m_{}{:d}".format("0" if _update_count < 10 else "", _update_count)
                mod_state_ids[state_id] = mod_state_id
                state_id = mod_state_id
                # update the state_id in the state_ids array

            topic_scores.append(np.sum(_temp_user_data["example_summaries"][state_id]["topic_dists"], axis=0).tolist())
            sbert_embeddings.append(np.mean(_temp_user_data["example_summaries"][state_id]["sbert_embeddings"], axis=0))
            summary_lengths.append(len(_temp_user_data["example_summaries"][state_id]["topic_dists"]))

        # Update the topic IDs for this set
        for key in mod_state_ids.keys():
            state_ids.remove(key)
            state_ids.append(mod_state_ids[key])
        
        topic_sums = topic_scores
        # print(topic_sums)

        # get the min and max of every topic from across all the summariess
        topic_min = list(np.min(topic_sums, axis=0))
        topic_max = list(np.max(topic_sums, axis=0))
        # avg_sum_len = avg_sum_len / len(example_summaires)

        # print(topic_min)
        # print(topic_max)

        # instantiate a numpy array to hold the generated topic bounds
        topic_query_bounds = np.zeros((self.nTopics_, 2))
        
        # I am not entirely sure why this and the rounding is here
        #     but this is the Vanillia SuDocu code
        eps = 1e-4
        
        for i in range(len(topic_min)):
            topic_query_bounds[i, 0] = round(topic_min[i][0] * 0.9 - eps, 4)
            topic_query_bounds[i, 1] = round(topic_max[i][0] * 1.1 + eps, 4)

        # print(topic_query_bounds)

        assert topic_query_bounds.shape[0] == self.nTopics_, "query bounds are not nTopics Long: SuDocuBase -> get_bounds"

        pref_data["state_ids"] = state_ids
        pref_data["topic_bounds"] = topic_query_bounds.tolist()
        pref_data["topic_scores"] = topic_scores
        pref_data["sbert_embeddings"] = sbert_embeddings
        pref_data["avg_summ_len"] = (float)(np.mean(summary_lengths))

        _temp_user_data["learned_preferences"]["preference_stats"].append(pref_data)

    
    def _extract_example_summary_data(self, _temp_user_data, data_entry):
        # dictionary to hold example data
        exam_entry_data = json.loads(",".join(data_entry[4:]))
        state_id = exam_entry_data["state_id"]

        # check if we have processed this state before
        if state_id in _temp_user_data["example_summaries"]["used_keys"]:
            # generate a modified state if
            _change_count = _temp_user_data["example_summaries"]["summary_update_counts"][state_id]
            modified_state_id = str(state_id) + "_m_{}{:d}".format("0" if _change_count < 10 else "", _change_count)
            # print(modified_state_id)
            _temp_user_data["example_summaries"]["summary_update_counts"][state_id] += 1
            # print(_temp_user_data["example_summaries"]["summary_update_counts"][state_id])
            
            # Now copy over the modified var for tracking            
            state_id = modified_state_id
            # while modified_state_id in temp_user_data["example_summaries"]["used_keys"]:
        else:
            # otherwise add this original file to the change counter
            _temp_user_data["example_summaries"]["summary_update_counts"][state_id] = 0

        target_doc = exam_entry_data["state_name"]
        # print("adding state_id: {:d}".format(state_id))

        # add the key
        _temp_user_data["example_summaries"]["used_keys"].add(state_id)

        # now use the sentence_ids to extract the actual state sentences / topic dists
        exam_entry_data["sentences"] = []
        exam_entry_data["topic_dists"] = []
        exam_entry_data["sbert_embeddings"] = []

        # load the NPZ file storing the SBERT embeddings for this document
        file_path = os.path.join(self.shared_docs_path_, "StateDocuments/", target_doc.strip() + "sudocu.npz")
        data = np.load(file_path)
        doc_sbert_embedding_data = data['embedding']

        # print("SBERT Target-Doc: ", target_doc)

        # document index offset
        test_doc_idx_offset = self.doc_indices[target_doc][0]
        
        # print("SBERT Target-Doc-Offset: ", test_doc_idx_offset)
        # print("SBERT Sentence-IDs: ", str(exam_entry_data["sentence_ids"]))
        
        for sentence_id in exam_entry_data["sentence_ids"]:
            exam_entry_data["sentences"].append(self.sentences[sentence_id])
            
            exam_entry_data["topic_dists"].append(self.df[self.df['sid'] == sentence_id][self.topic_names].to_numpy().transpose().tolist())

            exam_entry_data["sbert_embeddings"].append(doc_sbert_embedding_data[sentence_id - test_doc_idx_offset].tolist())

        # add the learned-prefernce number these examples are from (0-...)
        exam_entry_data["learned_pref_pointer"] =  _temp_user_data["learned_preferences"]["learned_count"]

        # then add the data to the dict of summaries, key-ed with state-id
        _temp_user_data["example_summaries"][state_id] = exam_entry_data


    def generate_sbert_graphs(self, user_ids, out_path=None, label_bars=False):
        # pull off the df to an easier to work-with variable (fuck ram honestly lol)
        _df = self.sbert_example_df
        # print(_df.columns)

        error_bar = "ci"

        if label_bars:
            error_bar = None

        # create the output directory
        base_path = os.getcwd() if out_path is None else out_path
        output_folder_path = os.path.join(base_path, "analyzed_data", "sbert")

        if not os.path.exists(output_folder_path):
            print(output_folder_path)
            os.makedirs(output_folder_path)

        for user in user_ids:

            print(user)
            # directory maintenance
            user_output_path = os.path.join(output_folder_path, user)

            if not os.path.exists(user_output_path):
                os.makedirs(user_output_path)
                os.makedirs(os.path.join(user_output_path, "examples_meta"))
                os.makedirs(os.path.join(user_output_path, "examples_meta", "sbert_scores"))
                os.makedirs(os.path.join(user_output_path, "examples_meta", "topic_score_diffs"))
                os.makedirs(os.path.join(user_output_path, "examples_meta", "rouge_scores"))

            ###
            ##
            ##  Topic-Score diffs
            ##
            ###
            
            # now do some meta-data graphs
            #     topic-score diffs averages, violins, and boxes
            total_ts_diff_avg = np.mean(_df[_df['user_id'] == user][self.topic_names].to_numpy())

            ax = sns.barplot(data=_df[(_df['user_id'] == user)][self.topic_names], estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_title(user + " T-S Diffs All (avg - {:.4f})".format(total_ts_diff_avg))
            if label_bars:
                for i in ax.containers:
                    ax.bar_label(i,)
                labels = ax.containers[0].datavalues
                ax.set_xlim(min(labels)*2, max(labels)*2)
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "avg_example_topic_score_all.png"))
            plt.close()

            # violins
            ax = sns.violinplot(data=_df[(_df['user_id'] == user)][self.topic_names], orient="h")
            ax.set_title(user + " T-S Diffs All (avg - {:.4f})".format(total_ts_diff_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "example_topic_score_dist_violin_all.png"))
            plt.close()

            # box-plots
            ax = sns.boxplot(data=_df[(_df['user_id'] == user)][self.topic_names], orient="h")
            ax.set_title(user + " T-S Diffs All (avg - {:.4f})".format(total_ts_diff_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "example_topic_score_dist_box_all.png"))
            plt.close()
            # end topic-score meta stuff


            ###
            ##
            ##  SBERT-Score
            ##
            ###
            
            #   viloins / boxes for SBERT scores
            # find total sums
            total_sbert_avg = np.mean(_df[_df['user_id'] == user]["cosine"].to_numpy()) 

            # total average plots
            g = sns.violinplot(data=_df[_df['user_id'] == user], y="cosine")
            g.set_title(user + " SBERT (Ex. v. Pred) (avg - {:.4f})".format(total_sbert_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "sbert_scores", "sbert_dists_violin.png"))
            plt.close()

            g = sns.boxplot(data=_df[_df['user_id'] == user], y="cosine")
            g.set_title(user + " SBERT (Ex. v. Pred) (avg - {:.4f})".format(total_sbert_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "sbert_scores", "sbert_dists_box.png"))
            plt.close()

            ###
            ##
            ##  Rouge-Scores
            ##
            ###

            # rouge scores:
            #    collect the averages/stds and barhcart it baby!
            #  also split by relaxation type
            raw_rouge_data = _df[self.rouge_scores][_df['user_id'] == user].to_numpy()
            averged_data = np.reshape(np.mean(raw_rouge_data, axis=0), newshape=(3, 4))
            stddev_data = np.reshape(np.std(raw_rouge_data, axis=0), newshape=(3, 4))

            x_labels = ["P", "R", "F1", "F2"]
            x_axis = [i for i in range(0, len(x_labels))]

            fig = plt.figure()
            ax1 = fig.subplots(1, 1)
            ax1.bar(x = x_axis, height = averged_data[0], yerr=stddev_data[0], label="Rouge-1", alpha=1.0)
            ax1.bar(x = x_axis, height = averged_data[2], yerr=stddev_data[2], label="Rouge-L", alpha=1.0)
            ax1.bar(x = x_axis, height = averged_data[1], yerr=stddev_data[1], label="Rouge-2", alpha=1.0)
            ax1.set_xticks(x_axis, x_labels)
            ax1.set_xlabel("Rouge IR-Metric")
            ax1.set_ylabel("Rouge Score Average (+/- STD)")
            ax1.set_title(user + " Average Rouge Scores")
            ax1.grid(True)
            ax1.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(user_output_path, "examples_meta", "rouge_scores", "rouge_scores_averages_all.png"))
            plt.close(fig)
            # end rouge scores

        # Do the meta-analysis across the users
        meta_analysis_folder = os.path.join(output_folder_path, "meta")


        if not os.path.exists(meta_analysis_folder):
            os.makedirs(meta_analysis_folder)
            os.makedirs(os.path.join(meta_analysis_folder, "rouge_scores"))
            os.makedirs(os.path.join(meta_analysis_folder, "rouge_scores", "per_user"))
            os.makedirs(os.path.join(meta_analysis_folder, "sbert_scores"))
            os.makedirs(os.path.join(meta_analysis_folder, "sbert_scores", "per_user"))
            os.makedirs(os.path.join(meta_analysis_folder, "topic_score_diffs"))

        ###
        ##
        ## some per-user meta
        ##
        ###

        # do all the rouge-scores
        for r_met_ in self.rouge_scores:
            # Calculate averages
            r_met_avg_all = np.mean(_df[r_met_].to_numpy())

            # averages
            ax = sns.barplot(data=_df, x=r_met_, y="user_id", estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_xlim(0, 1.0)
            ax.set_title("R.S. User-Avgs (All)({:.4f})".format(r_met_avg_all))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "avg_{}_score_all.png".format(r_met_)))
            plt.close()

            # violins
            ax = sns.violinplot(data=_df, x=r_met_, y="user_id", orient="h")
            ax.set_title("R.S. User-Avgs (All)({:.4f})".format(r_met_avg_all))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "violin_{}_score_all.png".format(r_met_)))
            plt.close()

            # boxes
            ax = sns.boxplot(data=_df, x=r_met_, y="user_id", orient="h")
            ax.set_title("R.S. User-Avgs (All)({:.4f})".format(r_met_avg_all))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "box_{}_score_all.png".format(r_met_)))
            plt.close()

        #  END ROUGE-SCORES


        # SBERT SCORES
        sbert_avg_all = np.mean(_df["cosine"].to_numpy())
        # averages
        ax = sns.barplot(data=_df, x="cosine", y="user_id", estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_xlim(0, 1.0)
        ax.set_title("SBERT User-Avgs (All)({:.4f})".format(sbert_avg_all))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "avg_sbert_score_all.png"))
        plt.close()

        # violins
        ax = sns.violinplot(data=_df, x="cosine", y="user_id", orient="h")
        ax.set_title("SBERT User-Avgs (All)({:.4f})".format(sbert_avg_all))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "violin_sbert_score_all.png"))
        plt.close()

        # boxes
        ax = sns.boxplot(data=_df, x="cosine", y="user_id", orient="h")
        ax.set_title("SBERT User-Avgs (All)({:.4f})".format(sbert_avg_all))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "box_sbert_score_all.png"))
        plt.close()
        # END SBERT SCORES

        ###
        ##
        ## END per-user meta
        ##
        ###

        ###
        ##
        ## Begin System-Level Analysis
        ##
        ###

        # Rouge Overall
        # rouge scores:
        #    collect the averages/stds and barhcart it baby!
        #  also split by relaxation type
        raw_rouge_data = _df[self.rouge_scores].to_numpy()
        averged_data = np.reshape(np.mean(raw_rouge_data, axis=0), newshape=(3, 4))
        stddev_data = np.reshape(np.std(raw_rouge_data, axis=0), newshape=(3, 4))

        x_labels = ["P", "R", "F1", "F2"]
        x_axis = [i for i in range(0, len(x_labels))]

        fig = plt.figure()
        ax1 = fig.subplots(1, 1)
        ax1.bar(x = x_axis, height = averged_data[0], yerr=stddev_data[0], label="Rouge-1", alpha=1.0)
        ax1.bar(x = x_axis, height = averged_data[2], yerr=stddev_data[2], label="Rouge-L", alpha=1.0)
        ax1.bar(x = x_axis, height = averged_data[1], yerr=stddev_data[1], label="Rouge-2", alpha=1.0)
        ax1.set_xticks(x_axis, x_labels)
        ax1.set_xlabel("Rouge IR-Metric")
        ax1.set_ylabel("Rouge Score Average (+/- STD)")
        ax1.set_title("Average Rouge Scores")
        ax1.grid(True)
        ax1.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "rouge_scores_averages_all.png"))
        plt.close(fig)
        # end rouge scores
        # END Rouge Overall

        # SBERT Overall
        # SBERT SCORES
        sbert_avg_all = np.mean(_df["cosine"].to_numpy())

        # averages
        ax = sns.barplot(data=_df, x="cosine", estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_xlim(0, 1.0)
        ax.set_title("SBERT User-Avgs (All)({:.4f})".format(sbert_avg_all))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "avg_sbert_score_all.png"))
        plt.close()
        ##############################################################

        # Topic-Score Diffs ###########
        ts_avg_all = np.mean(_df[self.topic_names].to_numpy())
        # averages
        ax = sns.barplot(data=_df[self.topic_names], estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_title("T.S. Diffs (all) ({:.4f})".format(ts_avg_all))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "avg_ts_diff_all.png"))
        plt.close()

        # violins 
        ax = sns.violinplot(data=_df[self.topic_names], orient="h")
        ax.set_title("T.S. Diffs (All) ({:.4f})".format(ts_avg_all))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "violin_ts_diff_all.png"))
        plt.close()

        # boxes
        ax = sns.boxplot(data=_df[self.topic_names], orient="h")
        ax.set_title("T.S. Diffs (All) ({:.4f})".format(ts_avg_all))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "box_ts_diff_all.png"))
        plt.close()

        # END Topic-Score Diffs ###########


    def generate_sudocu_graphs(self, user_ids, out_path=None, label_bars=False):
        # pull off the df to an easier to work-with variable (fuck ram honestly lol)
        _df = self.sudocu_sample_df
        # print(_df.columns)

        error_bar = "ci"

        if label_bars:
            error_bar = None

        # create the output directory
        base_path = os.getcwd() if out_path is None else out_path
        output_folder_path = os.path.join(base_path, "analyzed_data", "sudocu")
        # output_folder_path = os.path.join(base_path, "analyzed_data_final", "sudocu")

        if not os.path.exists(output_folder_path):
            print(output_folder_path)
            os.makedirs(output_folder_path)

        for user in user_ids:

            print(user)
            # directory maintenance
            user_output_path = os.path.join(output_folder_path, user)

            if not os.path.exists(user_output_path):
                os.makedirs(user_output_path)
                os.makedirs(os.path.join(user_output_path, "examples_meta"))
                os.makedirs(os.path.join(user_output_path, "examples_meta", "sbert_scores"))
                os.makedirs(os.path.join(user_output_path, "examples_meta", "topic_bound_diffs"))
                os.makedirs(os.path.join(user_output_path, "examples_meta", "topic_score_diffs"))
                os.makedirs(os.path.join(user_output_path, "examples_meta", "rouge_scores"))
                os.makedirs(os.path.join(user_output_path, "examples_meta", "slider_changes"))

            ###
            ##
            ##  Topic-Score diffs
            ##
            ###
            
            # now do some meta-data graphs
            #     topic-score diffs averages, violins, and boxes
            total_ts_diff_avg = np.mean(_df[_df['user_id'] == user][self.topic_names].to_numpy())
            doc_ts_diff_avg = np.mean(_df[(_df['user_id'] == user) & (_df["relax_type"] == "Doc")][self.topic_names].to_numpy())
            slider_ts_diff_avg = np.mean(_df[(_df['user_id'] == user) & (_df["relax_type"] == "Slider")][self.topic_names].to_numpy())

            ax = sns.barplot(data=_df[(_df['user_id'] == user)][self.topic_names], estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_title(user + " T-S Diffs All (avg - {:.4f},{:.4f},{:.4f})".format(total_ts_diff_avg, doc_ts_diff_avg, slider_ts_diff_avg))
            if label_bars:
                for i in ax.containers:
                    ax.bar_label(i,)
                labels = ax.containers[0].datavalues
                ax.set_xlim(min(labels)*2, max(labels)*2)
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "avg_example_topic_score_all.png"))
            plt.close()

            ax = sns.barplot(data=_df[(_df['user_id'] == user) & (_df["relax_type"] == "Doc")][self.topic_names], estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_title(user + " T-S Diffs Ex (avg - {:.4f},{:.4f},{:.4f})".format(total_ts_diff_avg, doc_ts_diff_avg, slider_ts_diff_avg))
            if label_bars:
                for i in ax.containers:
                    ax.bar_label(i,)
                labels = ax.containers[0].datavalues
                ax.set_xlim(min(labels)*2, max(labels)*2)
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "avg_example_topic_score_doc.png"))
            plt.close()

            ax = sns.barplot(data=_df[(_df['user_id'] == user) & (_df["relax_type"] == "Slider")][self.topic_names], estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_title(user + " T-S Diffs Re (avg - {:.4f},{:.4f},{:.4f})".format(total_ts_diff_avg, doc_ts_diff_avg, slider_ts_diff_avg))
            if label_bars:
                for i in ax.containers:
                    ax.bar_label(i,)
                labels = ax.containers[0].datavalues
                ax.set_xlim(min(labels)*2, max(labels)*2)
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "avg_example_topic_score_relax.png"))
            plt.close()

            # violins
            ax = sns.violinplot(data=_df[(_df['user_id'] == user)][self.topic_names], orient="h")
            ax.set_title(user + " T-S Diffs All (avg - {:.4f},{:.4f},{:.4f})".format(total_ts_diff_avg, doc_ts_diff_avg, slider_ts_diff_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "example_topic_score_dist_violin_all.png"))
            plt.close()

            ax = sns.violinplot(data=_df[(_df['user_id'] == user) & (_df["relax_type"] == "Doc")][self.topic_names], orient="h")
            ax.set_title(user + " T-S Diffs Ex (avg - {:.4f},{:.4f},{:.4f})".format(total_ts_diff_avg, doc_ts_diff_avg, slider_ts_diff_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "example_topic_score_dist_violin_doc.png"))
            plt.close()

            ax = sns.violinplot(data=_df[(_df['user_id'] == user) & (_df["relax_type"] == "Slider")][self.topic_names], orient="h")
            ax.set_title(user + " T-S Diffs Re (avg - {:.4f},{:.4f},{:.4f})".format(total_ts_diff_avg, doc_ts_diff_avg, slider_ts_diff_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "example_topic_score_dist_violin_relax.png"))
            plt.close()

            # box-plots
            ax = sns.boxplot(data=_df[(_df['user_id'] == user)][self.topic_names], orient="h")
            ax.set_title(user + " T-S Diffs All (avg - {:.4f},{:.4f},{:.4f})".format(total_ts_diff_avg, doc_ts_diff_avg, slider_ts_diff_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "example_topic_score_dist_box_all.png"))
            plt.close()

            ax = sns.boxplot(data=_df[(_df['user_id'] == user) & (_df["relax_type"] == "Doc")][self.topic_names], orient="h")
            ax.set_title(user + " T-S Diffs Ex (avg - {:.4f},{:.4f},{:.4f})".format(total_ts_diff_avg, doc_ts_diff_avg, slider_ts_diff_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "example_topic_score_dist_box_doc.png"))
            plt.close()

            ax = sns.boxplot(data=_df[(_df['user_id'] == user) & (_df["relax_type"] == "Slider")][self.topic_names], orient="h")
            ax.set_title(user + " T-S Diffs Re (avg - {:.4f},{:.4f},{:.4f})".format(total_ts_diff_avg, doc_ts_diff_avg, slider_ts_diff_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_score_diffs", "example_topic_score_dist_box_relax.png"))
            plt.close()
            # end topic-score meta stuff


            ###
            ##
            ##  SBERT-Score
            ##
            ###
            
            #   viloins / boxes for SBERT scores
            # find total sums
            total_sbert_avg = np.mean(_df[_df['user_id'] == user]["cosine"].to_numpy()) 
            doc_sbert_avg = np.mean(_df[(_df['user_id'] == user) & (_df["relax_type"] == "Doc")]["cosine"].to_numpy()) 
            slider_sbert_avg = np.mean(_df[(_df['user_id'] == user) & (_df["relax_type"] == "Slider")]["cosine"].to_numpy())

            print(total_sbert_avg)
            print(doc_sbert_avg)
            print(slider_sbert_avg)

            # bars of actual scores

            # actually generate plots
            #     these plots are split between Doc and Slider relaxation types
            g = sns.violinplot(data=_df[_df['user_id'] == user], x="relax_type", y="cosine")
            g.set_title(user + " SBERT (Ex. v. Pred) (avg - {:.4f},{:.4f},{:.4f})".format(total_sbert_avg, doc_sbert_avg, slider_sbert_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "sbert_scores", "sbert_dists_v_type_violin.png"))
            plt.close()

            g = sns.boxplot(data=_df[_df['user_id'] == user], x="relax_type", y="cosine")
            g.set_title(user + " SBERT (Ex. v. Pred) (avg - {:.4f},{:.4f},{:.4f})".format(total_sbert_avg, doc_sbert_avg, slider_sbert_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "sbert_scores", "sbert_dists_v_type_box.png"))
            plt.close()

            # total average plots
            g = sns.violinplot(data=_df[_df['user_id'] == user], y="cosine")
            g.set_title(user + " SBERT (Ex. v. Pred) (avg - {:.4f})".format(total_sbert_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "sbert_scores", "sbert_dists_violin.png"))
            plt.close()

            g = sns.boxplot(data=_df[_df['user_id'] == user], y="cosine")
            g.set_title(user + " SBERT (Ex. v. Pred) (avg - {:.4f})".format(total_sbert_avg))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "sbert_scores", "sbert_dists_box.png"))
            plt.close()

            ###
            ##
            ##  Topic-Bound diffs
            ##
            ###

        # Do we want these across relax-types as well??? 
            # bounds stuff
            #   extract the avg difference across all preds
            total_tb_diff_ex = np.mean(np.absolute(_df[_df['user_id'] == user][self.sum_bounds["example"]].to_numpy()))
            doc_tb_diff_ex = np.mean(np.absolute(_df[(_df['user_id'] == user)& (_df["relax_type"] == "Doc")][self.sum_bounds["example"]].to_numpy()))
            slider_tb_diff_ex = np.mean(np.absolute(_df[(_df['user_id'] == user) & (_df["relax_type"] == "Slider")][self.sum_bounds["example"]].to_numpy()))

            total_tb_diff_re = np.mean(np.absolute(_df[_df['user_id'] == user][self.sum_bounds["relax"]].to_numpy()))
            doc_tb_diff_re = np.mean(np.absolute(_df[(_df['user_id'] == user) & (_df["relax_type"] == "Doc")][self.sum_bounds["relax"]].to_numpy()))
            slider_tb_diff_re = np.mean(np.absolute(_df[(_df['user_id'] == user) & (_df["relax_type"] == "Slider")][self.sum_bounds["relax"]].to_numpy()))
            
            # basic bound-diffs avgs
            ax = sns.barplot(data=_df[self.all_bounds["example"]][_df['user_id'] == user], estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_title(user + " T-B Diffs (avg - {:.4f},{:.4f},{:.4f})".format(total_tb_diff_ex, doc_tb_diff_ex, slider_tb_diff_ex))
            if label_bars:
                for i in ax.containers:
                    ax.bar_label(i,)
                labels = ax.containers[0].datavalues
                ax.set_xlim(min(labels)*2, max(labels)*2)
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_bound_diffs", "avg_example_topic_bound_diffs_split.png"))
            plt.close()

            ax = sns.barplot(data=_df[self.all_bounds["relax"]][_df['user_id'] == user], estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_title(user + " T-B Diffs (avg - {:.4f},{:.4f},{:.4f})".format(total_tb_diff_re, doc_tb_diff_re, slider_tb_diff_re))
            if label_bars:
                for i in ax.containers:
                    ax.bar_label(i,)
                labels = ax.containers[0].datavalues
                ax.set_xlim(min(labels)*2, max(labels)*2)
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_bound_diffs", "avg_relax_topic_bound_diffs_split.png"))
            plt.close()

            ax = sns.barplot(data=_df[self.sum_bounds["example"]][_df['user_id'] == user], estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_title(user + " T-B Diffs (avg - {:.4f},{:.4f},{:.4f})".format(total_tb_diff_ex, doc_tb_diff_ex, slider_tb_diff_ex))
            if label_bars:
                for i in ax.containers:
                    ax.bar_label(i,)
                labels = ax.containers[0].datavalues
                ax.set_xlim(min(labels)*2, max(labels)*2)  
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_bound_diffs", "avg_example_topic_bound_diffs_sum.png"))
            plt.close()

            ax = sns.barplot(data=_df[self.sum_bounds["relax"]][_df['user_id'] == user], estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_title(user + " T-B Diffs (avg - {:.4f},{:.4f},{:.4f})".format(total_tb_diff_re, doc_tb_diff_re, slider_tb_diff_re))
            if label_bars:
                for i in ax.containers:
                    ax.bar_label(i,)
                labels = ax.containers[0].datavalues
                ax.set_xlim(min(labels)*2, max(labels)*2) 
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_bound_diffs", "avg_relax_topic_bound_diffs_sum.png"))
            plt.close()
            # end basic bound-diff avgs

            # start bound dists stuff
            ax = sns.violinplot(data=_df[self.sum_bounds["relax"]][_df['user_id'] == user], orient='h')
            ax.set_title(user + " T-B Diffs (avg - {:.4f},{:.4f},{:.4f})".format(total_tb_diff_re, doc_tb_diff_re, slider_tb_diff_re))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_bound_diffs", "relax_bound_diff_dists_violin.png"))
            plt.close()

            ax = sns.boxplot(data=_df[self.sum_bounds["relax"]][_df['user_id'] == user], orient='h')
            ax.set_title(user + " T-B Diffs (avg - {:.4f},{:.4f},{:.4f})".format(total_tb_diff_re, doc_tb_diff_re, slider_tb_diff_re))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_bound_diffs", "relax_bound_diff_dists_box.png"))
            plt.close()

            ax = sns.violinplot(data=_df[self.sum_bounds["example"]][_df['user_id'] == user], orient='h')
            ax.set_title(user + " T-B Diffs (avg - {:.4f},{:.4f},{:.4f})".format(total_tb_diff_ex, doc_tb_diff_ex, slider_tb_diff_ex))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_bound_diffs", "example_bound_diff_dists_violin.png"))
            plt.close()

            ax = sns.boxplot(data=_df[self.sum_bounds["example"]][_df['user_id'] == user], orient='h')
            ax.set_title(user + " T-B Diffs (avg - {:.4f},{:.4f},{:.4f})".format(total_tb_diff_ex, doc_tb_diff_ex, slider_tb_diff_ex))
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "topic_bound_diffs", "example_bound_diff_dists_box.png"))
            plt.close()
            # end bound-dists stuff


            ###
            ##
            ##  Rouge-Scores
            ##
            ###

            # rouge scores:
            #    collect the averages/stds and barhcart it baby!
            #  also split by relaxation type
            raw_rouge_data = _df[self.rouge_scores][_df['user_id'] == user].to_numpy()
            averged_data = np.reshape(np.mean(raw_rouge_data, axis=0), newshape=(3, 4))
            stddev_data = np.reshape(np.std(raw_rouge_data, axis=0), newshape=(3, 4))

            x_labels = ["P", "R", "F1", "F2"]
            x_axis = [i for i in range(0, len(x_labels))]

            fig = plt.figure()
            ax1 = fig.subplots(1, 1)
            ax1.bar(x = x_axis, height = averged_data[0], yerr=stddev_data[0], label="Rouge-1", alpha=1.0)
            ax1.bar(x = x_axis, height = averged_data[2], yerr=stddev_data[2], label="Rouge-L", alpha=1.0)
            ax1.bar(x = x_axis, height = averged_data[1], yerr=stddev_data[1], label="Rouge-2", alpha=1.0)
            ax1.set_xticks(x_axis, x_labels)
            ax1.set_xlabel("Rouge IR-Metric")
            ax1.set_ylabel("Rouge Score Average (+/- STD)")
            ax1.set_title(user + " Average Rouge Scores")
            ax1.grid(True)
            ax1.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(user_output_path, "examples_meta", "rouge_scores", "rouge_scores_averages_all.png"))
            plt.close(fig)
            # end rouge scores
            # now do the on-slider-change stuff

            # find the avergae about of change
            ax = sns.barplot(data=_df[["num_added", "num_removed", "total_change"]][(_df['user_id'] == user) & (_df["slider_mod"] == True)], estimator=np.mean, errorbar=error_bar, orient="h")
            ex_len_avg = "N/A"
            
            _vals =  _df[(_df['user_id'] == user) & (_df["slider_mod"] == True)]["evg_ex_len"].to_numpy()
            
            if len(_vals) < 1:
                ex_len_avg = np.mean(_vals)
                ax.set_title(user + " Slider-Change {:.4f}".format(ex_len_avg))
            else:
                ax.set_title(user + " Slider-Change {}".format(ex_len_avg))
            
            plt.tight_layout()
            plt.savefig(os.path.join(user_output_path, "examples_meta", "slider_changes", "avg_slider_change.png"))
            plt.close()


        # Do the meta-analysis across the users
        meta_analysis_folder = os.path.join(output_folder_path, "meta")


        if not os.path.exists(meta_analysis_folder):
            os.makedirs(meta_analysis_folder)
            os.makedirs(os.path.join(meta_analysis_folder, "rouge_scores"))
            os.makedirs(os.path.join(meta_analysis_folder, "rouge_scores", "per_user"))
            os.makedirs(os.path.join(meta_analysis_folder, "sbert_scores"))
            os.makedirs(os.path.join(meta_analysis_folder, "sbert_scores", "per_user"))
            os.makedirs(os.path.join(meta_analysis_folder, "topic_bound_diffs"))
            os.makedirs(os.path.join(meta_analysis_folder, "topic_bound_diffs", "example"))
            os.makedirs(os.path.join(meta_analysis_folder, "topic_bound_diffs", "relax"))
            os.makedirs(os.path.join(meta_analysis_folder, "topic_score_diffs"))

        ###
        ##
        ## some per-user meta
        ##
        ###

        # do all the rouge-scores
        for r_met_ in self.rouge_scores:
            # Calculate averages
            r_met_avg_all = np.mean(_df[r_met_].to_numpy())
            r_met_avg_do = np.mean(_df[_df["relax_type"] == "Doc"][r_met_].to_numpy())
            r_met_avg_re = np.mean(_df[_df["relax_type"] == "Slider"][r_met_].to_numpy())

            # averages
            ax = sns.barplot(data=_df, x=r_met_, y="user_id", estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_xlim(0, 1.0)
            ax.set_title("R.S. User-Avgs (All)({:.4f}, {:.4f}, {:.4f})".format(r_met_avg_all, r_met_avg_do, r_met_avg_re))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "avg_{}_score_all.png".format(r_met_)))
            plt.close()

            ax = sns.barplot(data=_df[_df["relax_type"] == "Doc"], x=r_met_, y="user_id", estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_xlim(0, 1.0)
            ax.set_title("R.S. User-Avgs (Doc)({:.4f}, {:.4f}, {:.4f})".format(r_met_avg_all, r_met_avg_do, r_met_avg_re))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "avg_{}_score_doc.png".format(r_met_)))
            plt.close()

            ax = sns.barplot(data=_df[_df["relax_type"] == "Slider"], x=r_met_, y="user_id", estimator=np.mean, errorbar=error_bar, orient="h")
            ax.set_xlim(0, 1.0)
            ax.set_title("R.S. User-Avgs (Re)({:.4f}, {:.4f}, {:.4f})".format(r_met_avg_all, r_met_avg_do, r_met_avg_re))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "avg_{}_score_relax.png".format(r_met_)))
            plt.close()

            # violins
            ax = sns.violinplot(data=_df, x=r_met_, y="user_id", orient="h")
            ax.set_title("R.S. User-Avgs (All)({:.4f}, {:.4f}, {:.4f})".format(r_met_avg_all, r_met_avg_do, r_met_avg_re))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "violin_{}_score_all.png".format(r_met_)))
            plt.close()

            ax = sns.violinplot(data=_df[_df["relax_type"] == "Doc"], x=r_met_, y="user_id", orient="h")
            ax.set_title("R.S. User-Avgs (Doc)({:.4f}, {:.4f}, {:.4f})".format(r_met_avg_all, r_met_avg_do, r_met_avg_re))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "violin_{}_score_doc.png".format(r_met_)))
            plt.close()

            ax = sns.violinplot(data=_df[_df["relax_type"] == "Slider"], x=r_met_, y="user_id", orient="h")
            ax.set_title("R.S. User-Avgs (Re)({:.4f}, {:.4f}, {:.4f})".format(r_met_avg_all, r_met_avg_do, r_met_avg_re))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "violin_{}_score_relax.png".format(r_met_)))
            plt.close()

            # boxes
            ax = sns.boxplot(data=_df, x=r_met_, y="user_id", orient="h")
            ax.set_title("R.S. User-Avgs (All)({:.4f}, {:.4f}, {:.4f})".format(r_met_avg_all, r_met_avg_do, r_met_avg_re))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "box_{}_score_all.png".format(r_met_)))
            plt.close()

            ax = sns.boxplot(data=_df[_df["relax_type"] == "Doc"], x=r_met_, y="user_id", orient="h")
            ax.set_title("R.S. User-Avgs (Doc)({:.4f}, {:.4f}, {:.4f})".format(r_met_avg_all, r_met_avg_do, r_met_avg_re))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "box_{}_score_doc.png".format(r_met_)))
            plt.close()

            ax = sns.boxplot(data=_df[_df["relax_type"] == "Slider"], x=r_met_, y="user_id", orient="h")
            ax.set_title("R.S. User-Avgs (Re)({:.4f}, {:.4f}, {:.4f})".format(r_met_avg_all, r_met_avg_do, r_met_avg_re))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "box_{}_score_relax.png".format(r_met_)))
            plt.close()


            # try some three-way plots
            ax = sns.swarmplot(data=_df, x=r_met_, y="user_id", hue="relax_type")
            ax.set_title("R.S. User-Avgs ({:.4f}, {:.4f}, {:.4f})".format(r_met_avg_all, r_met_avg_do, r_met_avg_re))
            plt.tight_layout()
            plt.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "per_user",  "swarm_{}_score_relax.png".format(r_met_)))
            plt.close()
        #
        #  END ROUGE-SCORES


        # SBERT SCORES
        sbert_avg_all = np.mean(_df["cosine"].to_numpy())
        sbert_met_avg_do = np.mean(_df[_df["relax_type"] == "Doc"]["cosine"].to_numpy())
        sbert_avg_re = np.mean(_df[_df["relax_type"] == "Slider"]["cosine"].to_numpy())
        # averages
        ax = sns.barplot(data=_df, x="cosine", y="user_id", estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_xlim(0, 1.0)
        ax.set_title("SBERT User-Avgs (All)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "avg_sbert_score_all.png"))
        plt.close()

        ax = sns.barplot(data=_df[_df["relax_type"] == "Doc"], x="cosine", y="user_id", estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_xlim(0, 1.0)
        ax.set_title("SBERT User-Avgs (Doc)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "avg_sbert_score_doc.png"))
        plt.close()

        ax = sns.barplot(data=_df[_df["relax_type"] == "Slider"], x="cosine", y="user_id", estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_xlim(0, 1.0)
        ax.set_title("SBERT User-Avgs (Re)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "avg_sbert_score_relax.png"))
        plt.close()

        # violins
        ax = sns.violinplot(data=_df, x="cosine", y="user_id", orient="h")
        ax.set_title("SBERT User-Avgs (All)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "violin_sbert_score_all.png"))
        plt.close()

        ax = sns.violinplot(data=_df[_df["relax_type"] == "Doc"], x="cosine", y="user_id", orient="h")
        ax.set_title("SBERT User-Avgs (Doc)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "violin_sbert_score_doc.png"))
        plt.close()

        ax = sns.violinplot(data=_df[_df["relax_type"] == "Slider"], x="cosine", y="user_id", orient="h")
        ax.set_title("SBERT User-Avgs (Re)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "violin_sbert_score_relax.png"))
        plt.close()

        # boxes
        ax = sns.boxplot(data=_df, x="cosine", y="user_id", orient="h")
        ax.set_title("SBERT User-Avgs (All)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "box_sbert_score_all.png"))
        plt.close()

        ax = sns.boxplot(data=_df[_df["relax_type"] == "Doc"], x="cosine", y="user_id", orient="h")
        ax.set_title("SBERT User-Avgs (Doc)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "box_sbert_score_doc.png"))
        plt.close()

        ax = sns.boxplot(data=_df[_df["relax_type"] == "Slider"], x="cosine", y="user_id", orient="h")
        ax.set_title("SBERT User-Avgs (Re)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "box_sbert_score_relax.png"))
        plt.close()
        

        # try some three-way plots
        ax = sns.swarmplot(data=_df, x="cosine", y="user_id", hue="relax_type")
        ax.set_title("SBERT User-Avgs ({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "per_user",  "swarm_sbert_score_relax.png"))
        plt.close()
        # END SBERT SCORES

        # add the user slider-change graph
        ax = sns.barplot(data=_df, x="total_change", y="user_id")
        ax.set_title("Avg. Total Slider Change")
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "avg_slider_change.png"))
        plt.close()

        ax = sns.barplot(data=_df, x="evg_ex_len", y="user_id")
        ax.set_title("Avg. Total Slider Change")
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "avg_exsumm_len.png"))
        plt.close()  

        ###
        ##
        ## END per-user meta
        ##
        ###

        ###
        ##
        ## Begin System-Level Analysis
        ##
        ###

        # Rouge Overall
        # rouge scores:
        #    collect the averages/stds and barhcart it baby!
        #  also split by relaxation type
        raw_rouge_data = _df[self.rouge_scores].to_numpy()
        averged_data = np.reshape(np.mean(raw_rouge_data, axis=0), newshape=(3, 4))
        stddev_data = np.reshape(np.std(raw_rouge_data, axis=0), newshape=(3, 4))

        x_labels = ["P", "R", "F1", "F2"]
        x_axis = [i for i in range(0, len(x_labels))]

        fig = plt.figure()
        ax1 = fig.subplots(1, 1)
        ax1.bar(x = x_axis, height = averged_data[0], yerr=stddev_data[0], label="Rouge-1", alpha=1.0)
        ax1.bar(x = x_axis, height = averged_data[2], yerr=stddev_data[2], label="Rouge-L", alpha=1.0)
        ax1.bar(x = x_axis, height = averged_data[1], yerr=stddev_data[1], label="Rouge-2", alpha=1.0)
        ax1.set_xticks(x_axis, x_labels)
        ax1.set_xlabel("Rouge IR-Metric")
        ax1.set_ylabel("Rouge Score Average (+/- STD)")
        ax1.set_title("Average Rouge Scores")
        ax1.grid(True)
        ax1.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "rouge_scores_averages_all.png"))
        plt.close(fig)

        raw_rouge_data = _df[_df["relax_type"] == "Doc"][self.rouge_scores].to_numpy()
        averged_data = np.reshape(np.mean(raw_rouge_data, axis=0), newshape=(3, 4))
        stddev_data = np.reshape(np.std(raw_rouge_data, axis=0), newshape=(3, 4))

        fig = plt.figure()
        ax1 = fig.subplots(1, 1)
        ax1.bar(x = x_axis, height = averged_data[0], yerr=stddev_data[0], label="Rouge-1", alpha=1.0)
        ax1.bar(x = x_axis, height = averged_data[2], yerr=stddev_data[2], label="Rouge-L", alpha=1.0)
        ax1.bar(x = x_axis, height = averged_data[1], yerr=stddev_data[1], label="Rouge-2", alpha=1.0)
        ax1.set_xticks(x_axis, x_labels)
        ax1.set_xlabel("Rouge IR-Metric")
        ax1.set_ylabel("Rouge Score Average (+/- STD)")
        ax1.set_title("Average Rouge Scores")
        ax1.grid(True)
        ax1.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "rouge_scores_averages_doc.png"))
        plt.close(fig)

        raw_rouge_data = _df[_df["relax_type"] == "Slider"][self.rouge_scores].to_numpy()
        averged_data = np.reshape(np.mean(raw_rouge_data, axis=0), newshape=(3, 4))
        stddev_data = np.reshape(np.std(raw_rouge_data, axis=0), newshape=(3, 4))

        fig = plt.figure()
        ax1 = fig.subplots(1, 1)
        ax1.bar(x = x_axis, height = averged_data[0], yerr=stddev_data[0], label="Rouge-1", alpha=1.0)
        ax1.bar(x = x_axis, height = averged_data[2], yerr=stddev_data[2], label="Rouge-L", alpha=1.0)
        ax1.bar(x = x_axis, height = averged_data[1], yerr=stddev_data[1], label="Rouge-2", alpha=1.0)
        ax1.set_xticks(x_axis, x_labels)
        ax1.set_xlabel("Rouge IR-Metric")
        ax1.set_ylabel("Rouge Score Average (+/- STD)")
        ax1.set_title("Average Rouge Scores")
        ax1.grid(True)
        ax1.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(meta_analysis_folder, "rouge_scores", "rouge_scores_averages_relax.png"))
        plt.close(fig)
        # end rouge scores
        # END Rouge Overall

        # SBERT Overall
        # SBERT SCORES
        sbert_avg_all = np.mean(_df["cosine"].to_numpy())
        sbert_met_avg_do = np.mean(_df[_df["relax_type"] == "Doc"]["cosine"].to_numpy())
        sbert_avg_re = np.mean(_df[_df["relax_type"] == "Slider"]["cosine"].to_numpy())

        # averages
        ax = sns.barplot(data=_df, x="cosine", estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_xlim(0, 1.0)
        ax.set_title("SBERT User-Avgs (All)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "avg_sbert_score_all.png"))
        plt.close()

        ax = sns.barplot(data=_df[_df["relax_type"] == "Doc"], x="cosine", estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_xlim(0, 1.0)
        ax.set_title("SBERT User-Avgs (Doc)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "avg_sbert_score_doc.png"))
        plt.close()

        ax = sns.barplot(data=_df[_df["relax_type"] == "Slider"], x="cosine", estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_xlim(0, 1.0)
        ax.set_title("SBERT User-Avgs (Re)({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "avg_sbert_score_relax.png"))
        plt.close()

        ax = sns.swarmplot(data=_df, x="cosine", hue="relax_type")
        ax.set_title("SBERT User-Avgs ({:.4f}, {:.4f}, {:.4f})".format(sbert_avg_all, sbert_met_avg_do, sbert_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "sbert_scores", "swarm_sbert_score_relax.png"))
        plt.close()
        # END SBERT overall
        ##############################################################

        # Topic-Score Diffs ###########
        ts_avg_all = np.mean(_df[self.topic_names].to_numpy())
        ts_met_avg_do = np.mean(_df[_df["relax_type"] == "Doc"][self.topic_names].to_numpy())
        ts_avg_re = np.mean(_df[_df["relax_type"] == "Slider"][self.topic_names].to_numpy())

        # averages
        ax = sns.barplot(data=_df[self.topic_names], estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_title("T.S. Diffs (all) ({:.4f}, {:.4f}, {:.4f})".format(ts_avg_all, ts_met_avg_do, ts_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "avg_ts_diff_all.png"))
        plt.close()

        ax = sns.barplot(data=_df[_df["relax_type"] == "Doc"][self.topic_names], estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_title("T.S. Diffs (Doc) ({:.4f}, {:.4f}, {:.4f})".format(ts_avg_all, ts_met_avg_do, ts_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "avg_ts_diff_doc.png"))
        plt.close()

        ax = sns.barplot(data=_df[_df["relax_type"] == "Slider"][self.topic_names], estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_title("T.S. Diffs (Relax) ({:.4f}, {:.4f}, {:.4f})".format(ts_avg_all, ts_met_avg_do, ts_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "avg_ts_diff_relax.png"))
        plt.close()

        # violins 
        ax = sns.violinplot(data=_df[self.topic_names], orient="h")
        ax.set_title("T.S. Diffs (All) ({:.4f}, {:.4f}, {:.4f})".format(ts_avg_all, ts_met_avg_do, ts_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "violin_ts_diff_all.png"))
        plt.close()

        ax = sns.violinplot(data=_df[_df["relax_type"] == "Doc"][self.topic_names], orient="h")
        ax.set_title("T.S. Diffs (Doc) ({:.4f}, {:.4f}, {:.4f})".format(ts_avg_all, ts_met_avg_do, ts_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "violin_ts_diff_doc.png"))
        plt.close()

        ax = sns.violinplot(data=_df[_df["relax_type"] == "Slider"][self.topic_names], orient="h")
        ax.set_title("T.S. Diffs (Relax) ({:.4f}, {:.4f}, {:.4f})".format(ts_avg_all, ts_met_avg_do, ts_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "violin_ts_diff_relax.png"))
        plt.close()

        # boxes
        ax = sns.boxplot(data=_df[self.topic_names], orient="h")
        ax.set_title("T.S. Diffs (All) ({:.4f}, {:.4f}, {:.4f})".format(ts_avg_all, ts_met_avg_do, ts_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "box_ts_diff_all.png"))
        plt.close()

        ax = sns.boxplot(data=_df[_df["relax_type"] == "Doc"][self.topic_names], orient="h")
        ax.set_title("T.S. Diffs (Doc) ({:.4f}, {:.4f}, {:.4f})".format(ts_avg_all, ts_met_avg_do, ts_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "box_ts_diff_doc.png"))
        plt.close()

        ax = sns.boxplot(data=_df[_df["relax_type"] == "Slider"][self.topic_names], orient="h")
        ax.set_title("T.S. Diffs (Relax) ({:.4f}, {:.4f}, {:.4f})".format(ts_avg_all, ts_met_avg_do, ts_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_score_diffs", "box_ts_diff_relax.png"))
        plt.close()

        # END Topic-Score Diffs ###########

        # BEGIN Example topic-bound diffs
        tb_avg_all = np.mean(_df[self.sum_bounds["example"]].to_numpy())
        tb_avg_do = np.mean(_df[_df["relax_type"] == "Doc"][self.sum_bounds["example"]].to_numpy())
        tb_avg_re = np.mean(_df[_df["relax_type"] == "Slider"][self.sum_bounds["example"]].to_numpy())

        # averages
        ax = sns.barplot(data=_df[self.sum_bounds["example"]], estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_title("T.B. Diffs (All) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "example", "avg_tb_diff_all.png"))
        plt.close()

        ax = sns.barplot(data=_df[_df["relax_type"] == "Doc"][self.sum_bounds["example"]], estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_title("T.B. Diffs (Doc) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "example", "avg_tb_diff_doc.png"))
        plt.close()

        ax = sns.barplot(data=_df[_df["relax_type"] == "Slider"][self.sum_bounds["example"]], estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_title("T.B. Diffs (Relax) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "example", "avg_tb_diff_relax.png"))
        plt.close()
        
        # violins 
        ax = sns.violinplot(data=_df[self.sum_bounds["example"]], orient="h")
        ax.set_title("T.B. Diffs (All) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "example", "violin_tb_diff_all.png"))
        plt.close()

        ax = sns.violinplot(data=_df[_df["relax_type"] == "Doc"][self.sum_bounds["example"]], orient="h")
        ax.set_title("T.B. Diffs (Doc) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "example", "violin_tb_diff_doc.png"))
        plt.close()

        ax = sns.violinplot(data=_df[_df["relax_type"] == "Slider"][self.sum_bounds["example"]], orient="h")
        ax.set_title("T.B. Diffs (Relax) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "example", "violin_tb_diff_relax.png"))
        plt.close()

        # boxes
        ax = sns.boxplot(data=_df[self.sum_bounds["example"]], orient="h")
        ax.set_title("T.B. Diffs (All) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "example", "box_tb_diff_all.png"))
        plt.close()

        ax = sns.boxplot(data=_df[_df["relax_type"] == "Doc"][self.sum_bounds["example"]], orient="h")
        ax.set_title("T.B. Diffs (Doc) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "example", "box_tb_diff_doc.png"))
        plt.close()

        ax = sns.boxplot(data=_df[_df["relax_type"] == "Slider"][self.sum_bounds["example"]], orient="h")
        ax.set_title("T.B. Diffs (Relax) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "example", "box_tb_diff_relax.png"))
        plt.close()
        # END Example topic-bound diffs       


        # BEGIN Relax topic-bound diffs
        tb_avg_all = np.mean(_df[self.sum_bounds["relax"]].to_numpy())
        tb_avg_do = np.mean(_df[_df["relax_type"] == "Doc"][self.sum_bounds["relax"]].to_numpy())
        tb_avg_re = np.mean(_df[_df["relax_type"] == "Slider"][self.sum_bounds["relax"]].to_numpy())

        # averages
        ax = sns.barplot(data=_df[self.sum_bounds["relax"]], estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_title("T.B. Diffs (All) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "relax", "avg_tb_diff_all.png"))
        plt.close()

        ax = sns.barplot(data=_df[_df["relax_type"] == "Doc"][self.sum_bounds["relax"]], estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_title("T.B. Diffs (Doc) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "relax", "avg_tb_diff_doc.png"))
        plt.close()

        ax = sns.barplot(data=_df[_df["relax_type"] == "Slider"][self.sum_bounds["relax"]], estimator=np.mean, errorbar=error_bar, orient="h")
        ax.set_title("T.B. Diffs (Relax) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "relax", "avg_tb_diff_relax.png"))
        plt.close()
        
        # violins 
        ax = sns.violinplot(data=_df[self.sum_bounds["relax"]], orient="h")
        ax.set_title("T.B. Diffs (All) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "relax", "violin_tb_diff_all.png"))
        plt.close()

        ax = sns.violinplot(data=_df[_df["relax_type"] == "Doc"][self.sum_bounds["relax"]], orient="h")
        ax.set_title("T.B. Diffs (Doc) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "relax", "violin_tb_diff_doc.png"))
        plt.close()

        ax = sns.violinplot(data=_df[_df["relax_type"] == "Slider"][self.sum_bounds["relax"]], orient="h")
        ax.set_title("T.B. Diffs (Relax) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "relax", "violin_tb_diff_relax.png"))
        plt.close()

        # boxes
        ax = sns.boxplot(data=_df[self.sum_bounds["relax"]], orient="h")
        ax.set_title("T.B. Diffs (All) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "relax", "box_tb_diff_all.png"))
        plt.close()

        ax = sns.boxplot(data=_df[_df["relax_type"] == "Doc"][self.sum_bounds["relax"]], orient="h")
        ax.set_title("T.B. Diffs (Doc) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "relax", "box_tb_diff_doc.png"))
        plt.close()

        ax = sns.boxplot(data=_df[_df["relax_type"] == "Slider"][self.sum_bounds["relax"]], orient="h")
        ax.set_title("T.B. Diffs (Relax) ({:.4f}, {:.4f}, {:.4f})".format(tb_avg_all, tb_avg_do, tb_avg_re))
        plt.tight_layout()
        plt.savefig(os.path.join(meta_analysis_folder, "topic_bound_diffs", "relax", "box_tb_diff_relax.png"))
        plt.close()
        # END Relax topic-bound diffs


    def generate_compariative_graphs(self, out_path=None, label_bars=False):
        pass



# Graphs:

#  Sudocu:
#    runtimes?????
#    For each user:
#      for each generated summary:
#         Generated R-scores (bar per-type, x-axis is P, R, F1 idk)
#         Topic Bound difference pred v avg of examples and relaxation (sum in note) (graph w lower and upper per topic / graph w lower+upper per topic)
#             3-way scatter bound-num v. bound diff v. relax-type  (requires restructuring)
#         Topic-score diff (sum in note)
#             3-way topic-score-num v. topic-score-diff v relax-type (requires restructuring)
#    (NEW DATA)
#         If slider-relax :
#            Amount of change w/ slider (avg. example summary length in note)
#               bound-change (per-topic w/ total in note) slider req bounds - prev bounds diff / relaxed bounds - prev bounds diff
#               topic-score change (per-topic w/ total in note)  - prev summ.
#               summary sentence change (# added sentences and # dropped sentences as two bars)
#    (END NEW DATA)
#
#      ----Avg. pred-example topic-bound diff (bar example and relaxation) (sum in note) (graph w lower and upper per topic / graph w lower+upper per topic)
#          lines (upper, lower, mag)
#      ---- Avg. Topic score diff (bar, lines example and relaxation) (sums in note)
#      ---- Avg. rouge scores (bars)
#      ---- Avg. SBERT score vs. relaxation type (avg across both in note)
#      ---- SBERT diff Violin/Box plot
#      ---- Topic-Score diff Violin/Box plot
#      ---- Topic-Bound diff Violin/Box plot 
#      
#      (NEW DATA)
#      avg. Amount of change w/ slider (avg. example summary length in note)
#          bound-change (sum abs() per-topic w/ total in note) slider (req bounds - prev relaxed bounds) / (new relaxed bounds - prev relaxed bounds)
#          topic-score change (per-topic w/ total in note)  - prev summ. (done)
#          summary sentence change (# added sentences and # dropped sentences as two bars)
#      (END NEW DATA)


#  Meta Data:
#     -----Avg. Rouge-Scores per. User (Graph for each sub-rouge type containing bars / scatter P/R/F1)
#     ---Avg. SBERT scores per. User
#        ---3-way scatter user v. SBERT scores v. relax type?

#     --- Avg. Relaxed Bound diff (graph w lower and upper per topic / graph w lower+upper per topic)

#     --- Avg. Example Bound diff (graph w lower and upper per topic / graph w lower+upper per topic)

#     ---- Avg. generated topic-score diff (pred v avg. examples)

#     (NEW DATA!)
#     ---Avg. example length vs. slider change amount
#        ----histogram? Yes
#     (END NEW DATA)


# Meta:

# SBERT scores 
#   3-way scatter user v SBERT score v system
#   bars
# Topic-Score diff
#   bars
# Rouge-Scores
#   3-way scatter (user. v Rouge-score v. system) for each score-type/sub-type
#   bars

