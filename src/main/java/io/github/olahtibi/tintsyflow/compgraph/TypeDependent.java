package io.github.olahtibi.tintsyflow.compgraph;

import io.github.olahtibi.tintsyflow.compgraph.matrix.Matrix;

public class TypeDependent {

    public static <T> T initWithValue(T withShape, Double value) {
        if(withShape instanceof Double) {
            return (T)new Double(1.0d);
        }
        else if(withShape instanceof Matrix) {
            Matrix result = ((Matrix)withShape).clone();
            double[][] data = result.getData();
            for(int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = value;
                }
            }
            return (T)result;
        }
        else {
            throw new IllegalArgumentException("Unsupported type!");
        }
    }

    public static <T> T increase(T value, Object increaseWith) {
        if(value instanceof Double && increaseWith instanceof Double) {
            double a = (Double)value;
            double b = (Double)increaseWith;
            return (T)new Double(a + b);
        }
        else if(value instanceof Matrix && increaseWith instanceof Matrix) {
            Matrix result = ((Matrix)value);
            double[][] resultData = result.getData();
            double[][] incrementData = ((Matrix) increaseWith).getData();
            for(int i = 0; i < resultData.length; i++) {
                for (int j = 0; j < resultData[i].length; j++) {
                    resultData[i][j] += incrementData[i][j];
                }
            }
            return (T)result;
        }
        else {
            throw new IllegalArgumentException("Unsupported type!");
        }
    }

    public static <T> T applyLearningRate(T value, T grad, double learningRate) {
        if(value instanceof Double && grad instanceof Double) {
            double a = (Double)value;
            double b = (Double)grad;
            return (T)new Double(a - (learningRate * b));
        }
        else if(value instanceof Matrix && grad instanceof Matrix) {
            Matrix a = ((Matrix)grad).times(learningRate);
            return (T)((Matrix)value).minus(a);
        }
        else {
            throw new IllegalArgumentException("Unsupported type!");
        }
    }

}
