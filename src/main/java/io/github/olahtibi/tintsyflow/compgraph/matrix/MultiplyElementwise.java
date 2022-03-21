package io.github.olahtibi.tintsyflow.compgraph.matrix;

import io.github.olahtibi.tintsyflow.compgraph.Node;
import io.github.olahtibi.tintsyflow.compgraph.Operation;
import java.util.ArrayList;
import java.util.List;

public class MultiplyElementwise extends Operation<Matrix> {

    public MultiplyElementwise(Node<Matrix> node1, Node<Matrix> node2) {
        super(node1, node2);
    }

    @Override
    public Matrix compute() {
        Matrix copy = inputNodes.get(0).getOutput().clone();
        double dataFirst[][] = copy.getData();
        double dataSecond[][] = inputNodes.get(1).getOutput().getData();
        for(int i = 0; i < dataFirst.length; i++) {
            for(int j = 0; j < dataFirst[i].length; j++) {
                dataFirst[i][j] *= dataSecond[i][j];
            }
        }
        return copy;
    }

    @Override
    public List<Matrix> computeDownstreamGradients(Matrix upstreamGradient) {
        List<Matrix> result = new ArrayList<>();
        result.add(inputNodes.get(0).getOutput().clone());
        result.add(inputNodes.get(1).getOutput().clone());
        double[][] dataFirst = result.get(0).getData();
        double[][] dataSecond = result.get(1).getData();
        for(int i = 0; i < dataFirst.length; i++) {
            for (int j = 0; j < dataFirst[i].length; j++) {
                dataFirst[i][j] = (upstreamGradient.getData()[i][j] * inputNodes.get(1).getOutput().getData()[i][j]);
            }
        }
        for(int i = 0; i < dataSecond.length; i++) {
            for (int j = 0; j < dataSecond[i].length; j++) {
                dataSecond[i][j] = (upstreamGradient.getData()[i][j] * inputNodes.get(0).getOutput().getData()[i][j]);
            }
        }
        return result;
    }

}
