package io.github.olahtibi.tintsyflow.compgraph;

import java.util.*;

public class Session<T> {

    private Map<Operation<T>, List<Node<T>>> traversals;

    public Session() {
        traversals = new HashMap<>();
    }

    public T run(Operation<T> operation, Map<Placeholder<T>, T> input) {
        List<Node<T>> traversal = lookupTraversal(operation);
        for(Node<T> node: traversal) {
            if(node instanceof Placeholder) {
                node.output = input.get(node);
            }
            else if(node instanceof Variable) {
                Variable<T> variable = (Variable<T>)node;
                node.output = variable.value;
            }
            else {
                node.output = ((Operation<T>)node).compute();
            }
        }
        return traversal.get(traversal.size() - 1).output;
    }

    public void gradientDescent(Operation<T> loss, double learningRate) {
        Map<Node<T>, T> gradTable = buildGradTable(loss);
        for(Node<T> node: buildGradTable(loss).keySet()) {
            if(node instanceof Variable) {
                T grad = gradTable.get(node);
                Variable variable = (Variable)node;
                variable.value = TypeDependent.applyLearningRate(variable.value, grad, learningRate);
            }
        }
    }

    public Map<Node<T>, T> buildGradTable(Operation<T> loss) {
        Map<Node<T>, T> gradTable = new LinkedHashMap<>();
        gradTable.put(loss, (T) TypeDependent.initWithValue(loss.output, 1.0d));
        HashSet<Node<T>> visited = new HashSet<>();
        Queue<Node<T>> queue  = new LinkedList<>();
        visited.add(loss);
        queue.add(loss);
        while(!queue.isEmpty()) {
            Node<T> node = queue.poll();
            if(node != loss) {
                gradTable.put(node, (T) TypeDependent.initWithValue(node.output, 0.0d));
                for(Operation<T> consumer: node.consumers) {
                    T lossGradWrtConsumerOutput = gradTable.get(consumer);
                    List<T> lossGradsWrtConsumerInputs = consumer.computeDownstreamGradients(lossGradWrtConsumerOutput);
                    int nodeIndexInConsumerInputs = consumer.inputNodes.indexOf(node);
                    T lossGradWrtNode = lossGradsWrtConsumerInputs.get(nodeIndexInConsumerInputs);
                    TypeDependent.increase(gradTable.get(node), lossGradWrtNode);
                }
            }
            if(node instanceof Operation) {
                List<Node<T>> inputNodes = ((Operation<T>)node).inputNodes;
                for(Node<T> inputNode: inputNodes) {
                    if(!visited.contains(inputNode)) {
                        visited.add(inputNode);
                        queue.add(inputNode);
                    }
                }
            }
        }
        return gradTable;
    }

    private List<Node<T>> lookupTraversal(Operation<T> operation) {
        List<Node<T>> traversal = traversals.get(operation);
        if(traversal == null) {
            traversal = postOrderTraversal(operation);
            traversals.put(operation, traversal);
        }
        return traversal;
    }

    private List<Node<T>> postOrderTraversal(Operation<T> operation) {
        List<Node<T>> traversal = new ArrayList<>();
        recursivePostOrderTraversal(operation, traversal);
        return traversal;
    }

    private void recursivePostOrderTraversal(Node<T> node, List<Node<T>> traversal) {
        if(node instanceof Operation) {
            Operation op = (Operation)node;
            List<Node<T>> inputNodes = op.inputNodes;
            for(Node<T> child: inputNodes) {
                recursivePostOrderTraversal(child, traversal);
            }
        }
        traversal.add(node);
    }

}
