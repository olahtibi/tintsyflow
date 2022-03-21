package io.github.olahtibi.tintsyflow.compgraph.matrix

import io.github.olahtibi.tintsyflow.compgraph.Operation
import io.github.olahtibi.tintsyflow.compgraph.Placeholder
import io.github.olahtibi.tintsyflow.compgraph.Session
import io.github.olahtibi.tintsyflow.compgraph.Variable
import spock.lang.Specification

class SingleLayerPerceptronTrainTest extends Specification {

    static double[][] bluePoints = [
        [3.53585077,  1.74743804],
        [2.08268017,  1.54089322],
        [4.14437353,  2.60863553],
        [2.15167766,  2.8214609 ]
    ]

    static double[][] redPoints = [
        [-1.37163481, -0.46140138],
        [-1.08078431, -1.66053298],
        [-0.34922356, -1.44996725],
        [-2.83225276, -2.86681762]
    ]

    /*
        Step 0: loss = 5.545177444479562
        Step 10: loss = 0.9534073755754825
        Step 20: loss = 0.5846488249186731
        Step 30: loss = 0.428931750739888
        Step 40: loss = 0.3406825975874936
        Step 50: loss = 0.2833038379722662
        Step 60: loss = 0.24282236019685985
        Step 70: loss = 0.2126555559149086
        Step 80: loss = 0.18927043126978377
        Step 90: loss = 0.17059220489021304
        Step 100: loss = 0.1553189185901869
    */
    def "should train single layer perceptron"() {
        given:
            Placeholder<Matrix> x = new Placeholder<>()
            Placeholder<Matrix> c = new Placeholder<>()
            HashMap<Placeholder<Matrix>, Matrix> feedDict = buildFeedDict(x, c);
            Variable<Matrix> w = new Variable<>(new Matrix((double[][])[
                [10.0, 10.0],
                [10.0, 10.0]
            ]))
            Variable<Matrix> b = new Variable<>(new Matrix((double[][])[
                [2.0, 2.0]
            ]))
            Operation<Matrix> probabilities = new Softmax(new AddRowVectorForEachRow(new Multiply(x, w), b));
            Operation<Matrix> nllLoss = new Negative(new ReduceSum(new ReduceSum(new MultiplyElementwise(c, new Log(probabilities)), 1)));
        when:
            Double lossValue
            Session<Matrix> session = new Session<>();
            for(int step = 0; step <= 100; step++) {
                lossValue = session.run(nllLoss, feedDict).getData()[0][0];
                // if(step % 10 == 0) {
                //     println("Step " + step + ": loss = " + lossValue);
                // }
                session.gradientDescent(nllLoss, 0.01d);
            }
        then:
            assert equals(lossValue, 0.15531d)
    }

    private static equals(double d1, double d2) {
        return Math.abs(d1 - d2) < 0.00001
    }

    private static HashMap<Placeholder<Matrix>, Matrix> buildFeedDict(Placeholder<Matrix> x, Placeholder<Matrix> c) {
        return new HashMap<Placeholder<Matrix>, Matrix>() {{
            put(x, buildX())
            put(c, buildC())
        }}
    }

    private static Matrix buildX() {
        double[][] data = new double[bluePoints.length + redPoints.length][2]
        for(int i = 0; i < bluePoints.length; i++) {
            data[i] = bluePoints[i]
        }
        for(int i = 0; i < redPoints.length; i++) {
            data[i + bluePoints.length] = redPoints[i]
        }
        return new Matrix(data)
    }

    private static Matrix buildC() {
        double[][] data = new double[bluePoints.length + redPoints.length][2]
        for(int i = 0; i < bluePoints.length; i++) {
            data[i] = (double[])[1.0d, 0.0d]
        }
        for(int i = 0; i < redPoints.length; i++) {
            data[i + bluePoints.length] = (double[])[0.0d, 1.0d]
        }
        return new Matrix(data);
    }

}
