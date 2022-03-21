package io.github.olahtibi.tintsyflow.compgraph.matrix;

import io.github.olahtibi.tintsyflow.compgraph.Node;
import io.github.olahtibi.tintsyflow.compgraph.Operation;
import java.util.ArrayList;
import java.util.List;

public class Negative extends Operation<Matrix> {

    public Negative(Node<Matrix> inputNode) {
        super(inputNode);
    }

    @Override
    public Matrix compute() {
        Matrix copy = inputNodes.get(0).getOutput().clone();
        double[][] data = copy.getData();
        for(int i = 0; i < data.length; i++) {
            for(int j = 0; j < data[i].length; j++) {
                data[i][j] = (0.0d - data[i][j]);
            }
        }
        return copy;
    }

    @Override
    public List<Matrix> computeDownstreamGradients(Matrix upstreamGradient) {
        List<Matrix> result = new ArrayList<>();
        result.add(inputNodes.get(0).getOutput().clone());
        double[][] data = result.get(0).getData();
        for(int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = (0.0d - upstreamGradient.getData()[i][j]);
            }
        }
        return result;
    }

}
