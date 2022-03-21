package io.github.olahtibi.tintsyflow.compgraph.matrix

import io.github.olahtibi.tintsyflow.compgraph.Placeholder
import io.github.olahtibi.tintsyflow.compgraph.Session
import io.github.olahtibi.tintsyflow.compgraph.Variable
import spock.lang.Specification

class AffineTransformationTest extends Specification {

    def "should calculate Ax+b"() {
        given:
            Placeholder<Matrix> x = new Placeholder<Matrix>()
            Variable<Matrix> A = new Variable<Matrix>(new Matrix((double[][])[
                [1.0, 0.0],
                [0.0, -1.0]]
            ))
            Variable<Matrix> b = new Variable<Matrix>(new Matrix((double[][])[
                [1.0],
                [1.0]]
            ))
            Multiply y = new Multiply(A, x)
            Add z = new Add(y, b)
        when:
            Session<Matrix> session = new Session<Matrix>()
            Matrix result = session.run(z, new HashMap<Placeholder<Matrix>, Matrix>() {{
                put(x, new Matrix((double[][])[
                    [1.0],
                    [2.0]]
                ))
            }})
        then:
            assert result.eq(new Matrix((double[][])[
               [2.0],
               [-1.0]
            ]))
    }

}
