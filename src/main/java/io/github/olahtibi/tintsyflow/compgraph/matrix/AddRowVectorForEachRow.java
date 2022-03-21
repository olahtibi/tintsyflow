package io.github.olahtibi.tintsyflow.compgraph.matrix;

import io.github.olahtibi.tintsyflow.compgraph.Node;
import io.github.olahtibi.tintsyflow.compgraph.Operation;
import java.util.ArrayList;
import java.util.List;

public class AddRowVectorForEachRow extends Operation<Matrix> {

    public AddRowVectorForEachRow(Node<Matrix> node1, Node<Matrix> node2) {
        super(node1, node2);
    }

    @Override
    public Matrix compute() {
        Matrix copy = inputNodes.get(0).getOutput().clone();
        Matrix m = inputNodes.get(1).getOutput();
        if(copy.getColumns() !=  m.getColumns()) {
            throw new IllegalArgumentException("Matrices are not compatible!");
        }
        if(m.getRows() != 1) {
            throw new IllegalArgumentException("Only row vectors are accepted!");
        }
        double[] rowVector = m.getData()[0];
        double[][] data = copy.getData();
        for(int j = 0; j < data.length; j++) {
            double[] row = data[j];
            for(int k = 0; k < row.length; k++) {
                data[j][k] = data[j][k] + rowVector[k];
            }
        }
        return copy;
    }

    @Override
    public List<Matrix> computeDownstreamGradients(Matrix upstreamGradient) {
        List<Matrix> result = new ArrayList<>();
        result.add(inputNodes.get(0).getOutput().clone());
        double[][] dataFirst = result.get(0).getData();
        for(int i = 0; i < dataFirst.length; i++) {
            for (int j = 0; j < dataFirst[i].length; j++) {
                dataFirst[i][j] = upstreamGradient.getData()[i][j];
            }
        }
        result.add(ReduceSum.compute(result.get(0), 0));
        return result;
    }

}
