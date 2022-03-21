package io.github.olahtibi.tintsyflow.compgraph.matrix;

import io.github.olahtibi.tintsyflow.compgraph.Node;
import io.github.olahtibi.tintsyflow.compgraph.Operation;
import java.util.ArrayList;
import java.util.List;

public class Softmax extends Operation<Matrix> {

    public Softmax(Node<Matrix> inputNode) {
        super(inputNode);
    }

    @Override
    public Matrix compute() {
        if(inputNodes.size() != 1) {
            throw new IllegalArgumentException("Only one argument is allowed!");
        }
        Matrix copy = inputNodes.get(0).getOutput().clone();
        double[][] data = copy.getData();
        for(int i = 0; i < data.length; i++) {
            double denominator = 0;
            for(int j = 0; j < data[i].length; j++) {
                denominator += Math.exp(data[i][j]);
            }
            for(int j = 0; j < data[i].length; j++) {
                data[i][j] = (Math.exp(data[i][j]) / denominator);
            }
        }
        return copy;
    }

    @Override
    public List<Matrix> computeDownstreamGradients(Matrix upstreamGradient) {
        List<Matrix> result = new ArrayList<>();
        result.add(output.clone());

        // Step 1: gradient * softmax
        double dataResult[][] = result.get(0).getData();
        double dataGrad[][] = upstreamGradient.getData();
        for(int i = 0; i < dataResult.length; i++) {
            for(int j = 0; j < dataResult[i].length; j++) {
                dataResult[i][j] *= dataGrad[i][j];
            }
        }

        // Step 2: transpose(reduce sum, axis = 1)
        Matrix reduceSum = ReduceSum.compute(result.get(0), 1).transpose();

        // Step 3: gradient - reduceSum
        for(int i = 0; i < dataResult.length; i++) {
            for(int j = 0; j < dataResult[i].length; j++) {
                dataResult[i][j] = dataGrad[i][j] - reduceSum.getData()[i][0];
            }
        }

        // Step 4. multiply by softmax
        double dataSoftmax[][] = output.getData();
        for(int i = 0; i < dataResult.length; i++) {
            for(int j = 0; j < dataResult[i].length; j++) {
                dataResult[i][j] *= dataSoftmax[i][j];
            }
        }

        return result;
    }

}
