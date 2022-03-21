package io.github.olahtibi.tintsyflow.compgraph.number;

import io.github.olahtibi.tintsyflow.compgraph.Node;
import io.github.olahtibi.tintsyflow.compgraph.Operation;
import java.util.ArrayList;
import java.util.List;

public class Multiply extends Operation<Double> {

    public Multiply(Node<Double> number1, Node<Double> number2) {
        super(number1, number2);
    }

    @Override
    public Double compute() {
        return inputNodes.get(0).getOutput() * inputNodes.get(1).getOutput();
    }

    @Override
    public List<Double> computeDownstreamGradients(Double upstreamGradient) {
        List<Double> result = new ArrayList<>();
        result.add(inputNodes.get(1).getOutput() * upstreamGradient);
        result.add(inputNodes.get(0).getOutput() * upstreamGradient);
        return result;
    }

}
