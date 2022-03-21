package io.github.olahtibi.tintsyflow.compgraph;

public class Variable<T> extends Node<T> {

    protected T value;

    public Variable(T value) {
        super();
        this.value = value;
    }

}
