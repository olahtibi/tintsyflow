package io.github.olahtibi.tintsyflow.compgraph.number

import io.github.olahtibi.tintsyflow.compgraph.Placeholder
import io.github.olahtibi.tintsyflow.compgraph.Session
import spock.lang.Specification

class NumberArithmeticTest extends Specification {

    def "should do simple arithmetic"() {
        given:
            Placeholder<Double> x = new Placeholder<Double>()
            Placeholder<Double> y = new Placeholder<Double>()
            Add z = new Add(x, y)
        when:
            Double result = new Session<>().run(z, new HashMap<Placeholder<Double>, Double>() {{
                put(x, 5.0d)
                put(y, 2.0d)
            }})
        then:
            assert result == 7.0d
    }

}
