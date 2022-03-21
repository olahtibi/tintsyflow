package io.github.olahtibi.tintsyflow;

import io.github.olahtibi.tintsyflow.compgraph.*;
import io.github.olahtibi.tintsyflow.compgraph.matrix.*;
import java.util.HashMap;

class MainProgram {

    static double[][] bluePoints = {
        { 3.53585077,  1.74743804},
        { 2.08268017,  1.54089322},
        { 4.14437353,  2.60863553},
        { 2.15167766,  2.8214609 }
    };

    static double[][] redPoints = {
        {-1.37163481, -0.46140138},
        {-1.08078431, -1.66053298},
        {-0.34922356, -1.44996725},
        {-2.83225276, -2.86681762}
    };

    public static void main(String args[]) {
        Placeholder<Matrix> x = new Placeholder<>();
        Placeholder<Matrix> c = new Placeholder<>();
        HashMap<Placeholder<Matrix>, Matrix> feedDict = buildFeedDict(x, c);
        Variable<Matrix> w = new Variable<>(new Matrix(new double[][] {
            {10.0, 10.0},
            {10.0, 10.0}
        }));
        Variable<Matrix> b = new Variable<>(new Matrix(new double[][] {
            {2.0, 2.0}
        }));
        Operation<Matrix> probabilities = new Softmax(new AddRowVectorForEachRow(new Multiply(x, w), b));
        Operation<Matrix> nllLoss = new Negative(new ReduceSum(new ReduceSum(new MultiplyElementwise(c, new Log(probabilities)), 1)));
        Session<Matrix> session = new Session<>();
        for(int step = 0; step <= 100; step++) {
            Double lossValue = session.run(nllLoss, feedDict).getData()[0][0];
            if(step % 10 == 0) {
                System.out.println("Step " + step + ": loss = " + lossValue);
            }
            session.gradientDescent(nllLoss, 0.01d);
        }
    }

    private static HashMap<Placeholder<Matrix>, Matrix> buildFeedDict(Placeholder<Matrix> x, Placeholder<Matrix> c) {
        return new HashMap<Placeholder<Matrix>, Matrix>() {{
            put(x, buildX());
            put(c, buildC());
        }};
    }

    private static Matrix buildX() {
        double[][] data = new double[bluePoints.length + redPoints.length][2];
        for(int i = 0; i < bluePoints.length; i++) {
            data[i] = bluePoints[i];
        }
        for(int i = 0; i < redPoints.length; i++) {
            data[i + bluePoints.length] = redPoints[i];
        }
        return new Matrix(data);
    }

    private static Matrix buildC() {
        double[][] data = new double[bluePoints.length + redPoints.length][2];
        for(int i = 0; i < bluePoints.length; i++) {
            data[i] = new double[] {1.0d, 0.0d};
        }
        for(int i = 0; i < redPoints.length; i++) {
            data[i + bluePoints.length] = new double[] {0.0d, 1.0d};
        }
        return new Matrix(data);
    }

} 