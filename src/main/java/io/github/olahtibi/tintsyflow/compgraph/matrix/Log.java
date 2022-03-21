package io.github.olahtibi.tintsyflow.compgraph.matrix;

import io.github.olahtibi.tintsyflow.compgraph.Node;
import io.github.olahtibi.tintsyflow.compgraph.Operation;
import java.util.ArrayList;
import java.util.List;

public class Log extends Operation<Matrix> {

    public Log(Node<Matrix> inputNode) {
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
                data[i][j] = Math.log(data[i][j]);
            }
        }
        return copy;
    }

    /*
    x = op.inputs[0]
    return grad/x
     */

    @Override
    public List<Matrix> computeDownstreamGradients(Matrix upstreamGradient) {
        List<Matrix> result = new ArrayList<>();
        result.add(inputNodes.get(0).getOutput().clone());
        double[][] data = result.get(0).getData();
        for(int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = (upstreamGradient.getData()[i][j] / data[i][j]);
            }
        }
        return result;
    }

}
