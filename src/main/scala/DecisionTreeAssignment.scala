
import org.apache.spark._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._


import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

object DecisionTreeAssignment {

  def main(args: Array[String]) {

    //spark context
    val sc = new SparkContext(new SparkConf().setAppName("Spark Word Count").setMaster("local"))

    //reading the input
    val rawData = sc.textFile("src/main/resources/covtype.data")

    val data = rawData.map { line =>
      val values = line.split(',').map(_.toDouble)
      val featureVector = Vectors.dense(values.init)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }

    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()


    //function for decision tree
    def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
      val predictionsAndLabels = data.map(example =>
        (model.predict(example.features), example.label)
      )
      new MulticlassMetrics(predictionsAndLabels)
    }

    val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), "gini", 4, 100)
    val metrics = getMetrics(model, cvData)


    val arr = metrics.confusionMatrix.toArray
    val total = arr.sum

    //to compute all p(i)
    val p = (0 to 6).map(i => (0 to 6).map(metrics.confusionMatrix(i, _)).sum / total)


    val overallPrecision = (0 to 6).map(i => metrics.precision(i) * p(i)).sum
    println("overallPrecision: " + overallPrecision)

    /*
    //this overall precision can also be found at
    metrics.weightedPrecision

    //find out pricision for each class
    metrics.precision(0)

    //find the confusion
    metrics.confusionMatrix
    */

  }

}
