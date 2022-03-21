package io.github.olahtibi.tintsyflow.compgraph.matrix;

import io.github.olahtibi.tintsyflow.compgraph.Node;
import io.github.olahtibi.tintsyflow.compgraph.Operation;
import java.util.ArrayList;
import java.util.List;

public class Sigmoid extends Operation<Matrix> {

    public Sigmoid(Node<Matrix> inputNode) {
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
            for(int j = 0; j < data[i].length; j++) {
                data[i][j] = sigmoid(data[i][j]);
            }
        }
        return copy;
    }

    @Override
    public List<Matrix> computeDownstreamGradients(Matrix upstreamGradient) {
        List<Matrix> result = new ArrayList<>();
        result.add(output.clone());
        double[][] data = result.get(0).getData();
        for(int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                double sigmoid = output.getData()[i][j];
                data[i][j] = (upstreamGradient.getData()[i][j] * ((1.0d - sigmoid) * sigmoid));
            }
        }
        return result;
    }

    private double sigmoid(double a) {
        return (1.0d / (1.0d + Math.exp(-a)));
    }

}
