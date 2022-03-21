package io.github.olahtibi.tintsyflow.compgraph.matrix

import io.github.olahtibi.tintsyflow.compgraph.Operation
import io.github.olahtibi.tintsyflow.compgraph.Session
import io.github.olahtibi.tintsyflow.compgraph.Variable
import spock.lang.Specification

class AddRowVectorForEachRowTest extends Specification {

    def "should compute gradients"() {
        given:
            Variable A = new Variable<Matrix>(new Matrix((double[][])[
                [1.0, 2.0],
                [3.0, 4.0]
            ]))
            Variable b = new Variable<Matrix>(new Matrix((double[][])[
                [5.0, 6.0]
            ]))
            Matrix uGrad = new Matrix((double[][])[
                [-1.0, 2.0],
                [-3.0, -4.0]
            ])
        when:
            Operation<Matrix> op = new AddRowVectorForEachRow(A, b)
            Matrix result = new Session<Matrix>().run(op, new HashMap())
            List<Matrix> dGrads = op.computeDownstreamGradients(uGrad)
        then:
            assert result.data == [[6.0, 8.0],[8.0, 10.0]]
            assert dGrads.size() == 2
            assert dGrads[0].data == [[-1.0, 2.0],[-3.0, -4.0]]
            assert dGrads[1].data == [[-4.0, -2.0]]
    }

}
