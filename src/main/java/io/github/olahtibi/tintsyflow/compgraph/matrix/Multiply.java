package io.github.olahtibi.tintsyflow.compgraph.matrix;

import io.github.olahtibi.tintsyflow.compgraph.Node;
import io.github.olahtibi.tintsyflow.compgraph.Operation;
import java.util.ArrayList;
import java.util.List;

public class Multiply extends Operation<Matrix> {

    public Multiply(Node<Matrix> node1, Node<Matrix> node2) {
        super(node1, node2);
    }

    @Override
    public Matrix compute() {
        return inputNodes.get(0).getOutput().times(inputNodes.get(1).getOutput());
    }

    @Override
    public List<Matrix> computeDownstreamGradients(Matrix upstreamGradient) {
        List<Matrix> result = new ArrayList<>();
        Matrix a = inputNodes.get(0).getOutput();
        Matrix b = inputNodes.get(1).getOutput();
        result.add(upstreamGradient.times(b.transpose()));
        result.add(a.transpose().times(upstreamGradient));
        return result;
    }

}
