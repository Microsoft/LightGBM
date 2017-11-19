#ifndef LIGHTGBM_BOOSTING_H_
#define LIGHTGBM_BOOSTING_H_

#include <LightGBM/meta.h>
#include <LightGBM/config.h>

#ifdef USE_PROTO
#include "model.pb.h"
#endif // USE_PROTO

#include <vector>
#include <string>
#include <map>

namespace LightGBM {

/*! \brief forward declaration */
class Dataset;
class ObjectiveFunction;
class Metric;
struct PredictionEarlyStopInstance;

/*!
* \brief The interface for Boosting
*/
class LIGHTGBM_EXPORT Boosting {
public:
  /*! \brief virtual destructor */
  virtual ~Boosting() {}

  /*!
  * \brief Initialization logic
  * \param config Configs for boosting
  * \param train_data Training data
  * \param objective_function Training objective function
  * \param training_metrics Training metric
  */
  virtual void Init(
    const BoostingConfig* config,
    const Dataset* train_data,
    const ObjectiveFunction* objective_function,
    const std::vector<const Metric*>& training_metrics) = 0;

  /*!
  * \brief Merge model from other boosting object
  Will insert to the front of current boosting object
  * \param other
  */
  virtual void MergeFrom(const Boosting* other) = 0;

  virtual void ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
                                 const std::vector<const Metric*>& training_metrics) = 0;

  virtual void ResetConfig(const BoostingConfig* config) = 0;

  /*!
  * \brief Add a validation data
  * \param valid_data Validation data
  * \param valid_metrics Metric for validation data
  */
  virtual void AddValidDataset(const Dataset* valid_data,
                               const std::vector<const Metric*>& valid_metrics) = 0;

  virtual void Train(int snapshot_freq, const std::string& model_output_path) = 0;

  /*!
  * \brief Training logic
  * \param gradients nullptr for using default objective, otherwise use self-defined boosting
  * \param hessians nullptr for using default objective, otherwise use self-defined boosting
  * \return True if cannot train anymore
  */
  virtual bool TrainOneIter(const score_t* gradients, const score_t* hessians) = 0;

  /*!
  * \brief Rollback one iteration
  */
  virtual void RollbackOneIter() = 0;

  /*!
  * \brief return current iteration
  */
  virtual int GetCurrentIteration() const = 0;

  /*!
  * \brief Get evaluation result at data_idx data
  * \param data_idx 0: training data, 1: 1st validation data
  * \return evaluation result
  */
  virtual std::vector<double> GetEvalAt(int data_idx) const = 0;

  /*!
  * \brief Get current training score
  * \param out_len length of returned score
  * \return training score
  */
  virtual const double* GetTrainingScore(int64_t* out_len) = 0;

  /*!
  * \brief Get prediction result at data_idx data
  * \param data_idx 0: training data, 1: 1st validation data
  * \return out_len length of returned score
  */
  virtual int64_t GetNumPredictAt(int data_idx) const = 0;

  /*!
  * \brief Get prediction result at data_idx data
  * \param data_idx 0: training data, 1: 1st validation data
  * \param result used to store prediction result, should allocate memory before call this function
  * \param out_len length of returned score
  */
  virtual void GetPredictAt(int data_idx, double* result, int64_t* out_len) = 0;

  virtual int NumPredictOneRow(int num_iteration, bool is_pred_leaf, bool is_pred_contrib) const = 0;

  /*!
  * \brief Prediction for one record, not sigmoid transform
  * \param feature_values Feature value on this record
  * \param output Prediction result for this record
  * \param early_stop Early stopping instance. If nullptr, no early stopping is applied and all models are evaluated.
  */
  virtual void PredictRaw(const double* features, double* output,
                          const PredictionEarlyStopInstance* early_stop) const = 0;

  virtual void PredictRawByMap(const std::unordered_map<int, double>& features, double* output,
                               const PredictionEarlyStopInstance* early_stop) const = 0;


  /*!
  * \brief Prediction for one record, sigmoid transformation will be used if needed
  * \param feature_values Feature value on this record
  * \param output Prediction result for this record
  * \param early_stop Early stopping instance. If nullptr, no early stopping is applied and all models are evaluated.
  */
  virtual void Predict(const double* features, double* output,
                       const PredictionEarlyStopInstance* early_stop) const = 0;

  virtual void PredictByMap(const std::unordered_map<int, double>& features, double* output,
                            const PredictionEarlyStopInstance* early_stop) const = 0;


  /*!
  * \brief Prediction for one record with leaf index
  * \param feature_values Feature value on this record
  * \param output Prediction result for this record
  */
  virtual void PredictLeafIndex(
    const double* features, double* output) const = 0;

  virtual void PredictLeafIndexByMap(
    const std::unordered_map<int, double>& features, double* output) const = 0;

  /*!
  * \brief Feature contributions for the model's prediction of one record
  * \param feature_values Feature value on this record
  * \param output Prediction result for this record
  * \param early_stop Early stopping instance. If nullptr, no early stopping is applied and all models are evaluated.
  */
  virtual void PredictContrib(const double* features, double* output,
                              const PredictionEarlyStopInstance* early_stop) const = 0;

  /*!
  * \brief Dump model to json format string
  * \param num_iteration Number of iterations that want to dump, -1 means dump all
  * \return Json format string of model
  */
  virtual std::string DumpModel(int num_iteration) const = 0;

  /*!
  * \brief Translate model to if-else statement
  * \param num_iteration Number of iterations that want to translate, -1 means translate all
  * \return if-else format codes of model
  */
  virtual std::string ModelToIfElse(int num_iteration) const = 0;

  /*!
  * \brief Translate model to if-else statement
  * \param num_iteration Number of iterations that want to translate, -1 means translate all
  * \param filename Filename that want to save to
  * \return is_finish Is training finished or not
  */
  virtual bool SaveModelToIfElse(int num_iteration, const char* filename) const = 0;

  /*!
  * \brief Save model to file
  * \param num_iterations Number of model that want to save, -1 means save all
  * \param is_finish Is training finished or not
  * \param filename Filename that want to save to
  * \return true if succeeded
  */
  virtual bool SaveModelToFile(int num_iterations, const char* filename) const = 0;

  /*!
  * \brief Save model to string
  * \param num_iterations Number of model that want to save, -1 means save all
  * \return Non-empty string if succeeded
  */
  virtual std::string SaveModelToString(int num_iterations) const = 0;

  /*!
  * \brief Restore from a serialized string
  * \param model_str The string of model
  * \return true if succeeded
  */
  virtual bool LoadModelFromString(const std::string& model_str) = 0;

  #ifdef USE_PROTO
  /*!
  * \brief Save model with protobuf
  * \param num_iterations Number of model that want to save, -1 means save all
  * \param filename Filename that want to save to
  */
  virtual void SaveModelToProto(int num_iteration, const char* filename) const = 0;

  /*!
  * \brief Restore from a serialized protobuf file
  * \param filename Filename that want to restore from
  * \return true if succeeded
  */
  virtual bool LoadModelFromProto(const char* filename) = 0;
  #endif // USE_PROTO

  /*!
  * \brief Calculate feature importances
  * \param num_iteration Number of model that want to use for feature importance, -1 means use all
  * \param importance_type: 0 for split, 1 for gain
  * \return vector of feature_importance
  */
  virtual std::vector<double> FeatureImportance(int num_iteration, int importance_type) const = 0;

  /*!
  * \brief Get max feature index of this model
  * \return Max feature index of this model
  */
  virtual int MaxFeatureIdx() const = 0;

  /*!
  * \brief Get feature names of this model
  * \return Feature names of this model
  */
  virtual std::vector<std::string> FeatureNames() const = 0;

  /*!
  * \brief Get index of label column
  * \return index of label column
  */
  virtual int LabelIdx() const = 0;

  /*!
  * \brief Get number of weak sub-models
  * \return Number of weak sub-models
  */
  virtual int NumberOfTotalModel() const = 0;

  /*!
  * \brief Get number of models per iteration
  * \return Number of models per iteration
  */
  virtual int NumModelPerIteration() const = 0;

  /*!
  * \brief Get number of classes
  * \return Number of classes
  */
  virtual int NumberOfClasses() const = 0;

  /*! \brief The prediction should be accurate or not. True will disable early stopping for prediction. */
  virtual bool NeedAccuratePrediction() const = 0;

  /*!
  * \brief Initial work for the prediction
  * \param num_iteration number of used iteration
  */
  virtual void InitPredict(int num_iteration) = 0;

  /*!
  * \brief Name of submodel
  */
  virtual const char* SubModelName() const = 0;

  Boosting() = default;
  /*! \brief Disable copy */
  Boosting& operator=(const Boosting&) = delete;
  /*! \brief Disable copy */
  Boosting(const Boosting&) = delete;

  static bool LoadFileToBoosting(Boosting* boosting, const std::string& format, const char* filename);

  /*!
  * \brief Create boosting object
  * \param type Type of boosting
  * \param format Format of model
  * \param config config for boosting
  * \param filename name of model file, if existing will continue to train from this model
  * \return The boosting object
  */
  static Boosting* CreateBoosting(const std::string& type, const std::string& format, const char* filename);

};

class GBDTBase : public Boosting {
public:
  virtual double GetLeafValue(int tree_idx, int leaf_idx) const = 0;
  virtual void SetLeafValue(int tree_idx, int leaf_idx, double val) = 0;
};

}  // namespace LightGBM

#endif   // LightGBM_BOOSTING_H_
