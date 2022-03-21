package io.github.olahtibi.tintsyflow.compgraph.matrix;

import io.github.olahtibi.tintsyflow.compgraph.Node;
import io.github.olahtibi.tintsyflow.compgraph.Operation;

import java.util.ArrayList;
import java.util.List;

public class ReduceSum extends Operation<Matrix> {

    private Integer axis;

    public ReduceSum(Node<Matrix> inputNode) {
        super(inputNode);
        this.axis = null;
    }

    public ReduceSum(Node<Matrix> inputNode, Integer axis) {
        super(inputNode);
        this.axis = axis;
    }

    @Override
    public Matrix compute() {
        if(inputNodes.size() != 1) {
            throw new IllegalArgumentException("Only one argument is allowed!");
        }
        return compute(inputNodes.get(0).getOutput(), axis);
    }

    public static Matrix compute(Matrix matrix, Integer axis) {
        double[][] data = matrix.getData();
        if(axis == null) {
            double sum = 0.0d;
            for(int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    sum += data[i][j];
                }
            }
            return new Matrix(new double[][]{{sum}});
        }
        else if(axis == 0) {
            Matrix result = Matrix.zero(1, data[0].length);
            for(int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    result.getData()[0][j] += data[i][j];
                }
            }
            return result;
        }
        else if(axis == 1) {
            Matrix result = Matrix.zero(1, data.length);
            for(int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    result.getData()[0][i] += data[i][j];
                }
            }
            return result;
        }
        else {
            throw new IllegalArgumentException("Axis must be null, 0 or 1!");
        }
    }

    @Override
    public List<Matrix> computeDownstreamGradients(Matrix upstreamGradient) {
        Matrix result = inputNodes.get(0).getOutput().clone();
        double[][] data = result.getData();
        if(axis == null) {
            for(int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = upstreamGradient.getData()[0][0];
                }
            }
        }
        else if(axis == 0) {
            for(int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = upstreamGradient.getData()[0][j];
                }
            }
        }
        else if(axis == 1) {
            for(int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = upstreamGradient.getData()[0][i];
                }
            }
        }
        List<Matrix> resultList = new ArrayList<>();
        resultList.add(result);
        return resultList;
    }

}
