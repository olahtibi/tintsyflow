package io.github.olahtibi.tintsyflow.compgraph.matrix

import io.github.olahtibi.tintsyflow.compgraph.Operation
import io.github.olahtibi.tintsyflow.compgraph.Session
import io.github.olahtibi.tintsyflow.compgraph.Variable
import spock.lang.Specification
import spock.lang.Unroll

class ReduceSumTest extends Specification {

    @Unroll
    def "should calculate reduce sum - axis = #axis"() {
        given:
            Variable A = new Variable<Matrix>(new Matrix((double[][])[
                [1.0, 2.0],
                [3.0, 4.0]
            ]))
        when:
            Matrix result = new Session<Matrix>().run(new ReduceSum(A, axis), new HashMap())
        then:
            assert result.data == resultData
        where:
            axis | resultData
            null | [[10.0]]
            0    | [[4.0, 6.0]]
            1    | [[3.0, 7.0]]
    }

    @Unroll
    def "should calculate gradient - axis = #axis"() {
        given:
            Variable A = new Variable<Matrix>(new Matrix((double[][])[
                [1.0, 2.0],
                [3.0, 4.0]
            ]))
        when:
            Operation<Matrix> op = new ReduceSum(A, axis)
            Matrix result = new Session<Matrix>().run(op, new HashMap())
            List<Matrix> dGrads = op.computeDownstreamGradients(new Matrix((double[][])uGrad))
        then:
            assert result.data == resultData
            assert new Matrix((double[][])expectedGrads).eq(dGrads)
        where:
            axis | resultData   | uGrad          | expectedGrads
            null | [[10.0]]     | [[-2.0]]       | [[-2.0, -2.0],[-2.0, -2.0]]
            0    | [[4.0, 6.0]] | [[-1.0, -2.0]] | [[-1.0, -2.0],[-1.0, -2.0]]
            1    | [[3.0, 7.0]] | [[-1.0, -2.0]] | [[-1.0, -1.0],[-2.0, -2.0]]
    }

}
