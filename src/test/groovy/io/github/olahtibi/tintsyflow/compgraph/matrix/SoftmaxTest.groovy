package io.github.olahtibi.tintsyflow.compgraph.matrix

import io.github.olahtibi.tintsyflow.compgraph.Operation
import io.github.olahtibi.tintsyflow.compgraph.Placeholder
import spock.lang.Specification

class SoftmaxTest extends Specification {

    def "should compute gradients"() {
        given:
            Matrix uGrad = new Matrix((double[][])[
                [-1.0, -2.0],
                [-3.0, 4.0],
                [5.0, -6.0]
            ])
        when:
            Operation<Matrix> op = new Softmax(new Placeholder<Matrix>())
            op.output = new Matrix((double[][])[
                [9.0, 1.0],
                [5.0, 5.0],
                [3.0, 7.0]
            ])
            List<Matrix> dGrads = op.computeDownstreamGradients(uGrad)
        then:
            assert dGrads.size() == 1
            assert dGrads[0].data == [[90.0, 9.0],[-40.0, -5.0],[96, 147]]
    }

}
