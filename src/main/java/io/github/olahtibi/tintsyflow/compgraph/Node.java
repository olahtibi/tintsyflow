package io.github.olahtibi.tintsyflow.compgraph;

import java.util.ArrayList;
import java.util.List;

// Represents a node in the computational graph
public class Node<T> {

    protected T output;
    protected List<Operation<T>> consumers;

    public Node() {
        this.consumers = new ArrayList<Operation<T>>();
    }

    public T getOutput() {
        return output;
    }

}
