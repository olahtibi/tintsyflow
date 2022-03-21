package io.github.olahtibi.tintsyflow.compgraph;

import java.util.Arrays;
import java.util.List;

public abstract class Operation<T> extends Node<T> {

    protected List<Node<T>> inputNodes;

    public Operation(Node<T>... inputNodes) {
        super();
        this.inputNodes = Arrays.asList(inputNodes);
        for(Node<T> inputNode: inputNodes) {
            inputNode.consumers.add(this);
        }
    }

    public abstract T compute();

    public abstract List<T> computeDownstreamGradients(T upstreamGradient);

}
