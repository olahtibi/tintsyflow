package io.github.olahtibi.tintsyflow.compgraph.matrix

import io.github.olahtibi.tintsyflow.compgraph.Placeholder
import io.github.olahtibi.tintsyflow.compgraph.Session
import io.github.olahtibi.tintsyflow.compgraph.Variable
import spock.lang.Specification

class BinaryClassifierTest extends Specification {

    static final Matrix bluePoints = new Matrix((double[][])[[ 3.09059949,  1.85866489],
                             [ 1.90234734,  1.41928718],
                             [ 2.76179057,  0.19321828],
                             [ 1.04084757,  3.12329185],
                             [ 4.15445356,  3.43075616],
                             [ 0.81293188,  2.98479123],
                             [ 1.44204811,  1.25870624],
                             [ 1.56369793,  0.75513533],
                             [ 2.08813833,  2.73753279],
                             [ 1.5735684 , -0.12554767],
                             [ 3.40278846,  1.79520929],
                             [ 2.22949943,  1.1412392 ],
                             [ 1.12564447,  3.27688185],
                             [ 2.25452207,  1.97040615],
                             [ 1.82985441,  2.06114495],
                             [ 0.16821953,  2.33085776],
                             [ 1.56749236,  1.88485612],
                             [ 1.27696369,  2.38513947],
                             [ 0.46914291,  2.44894365],
                             [ 1.72871082,  1.83625035],
                             [ 2.4210018 ,  2.31501015],
                             [ 1.27105769,  2.27002583],
                             [ 0.72313227,  1.90238237],
                             [ 1.49770912,  1.66066323],
                             [ 3.5653465 ,  1.36037566],
                             [ 2.25019212,  2.35482405],
                             [ 2.14245118,  1.03056471],
                             [ 2.8602002 ,  1.96454502],
                             [ 2.78380515,  2.11543055],
                             [ 1.57427895,  1.75645882],
                             [ 3.34742703, -0.70071509],
                             [ 1.02546506,  1.13126859],
                             [ 2.20948185,  2.41308209],
                             [ 2.68082349,  0.90136602],
                             [ 2.89774508,  2.9741704 ],
                             [ 1.99138606,  1.45674249],
                             [ 1.36697618,  3.43873337],
                             [ 1.35555301,  0.74870331],
                             [ 1.74543378,  1.45071702],
                             [ 2.33632292,  0.86645889],
                             [ 2.74921301,  1.88024432],
                             [ 1.77454685,  1.38626251],
                             [ 1.51631113,  1.75389786],
                             [ 2.96404871,  3.66805796],
                             [ 2.36139514,  1.4111184 ],
                             [ 4.00810097,  2.05473211],
                             [ 0.41073724,  0.52812644],
                             [ 0.75400166,  3.22752099],
                             [ 3.66998005,  2.24053604],
                             [ 3.59063233,  2.62541599]])

    static final Matrix redPoints = new Matrix((double[][])[[-1.77072003, -2.48181502],
                            [-2.34062954, -2.95462086],
                            [-1.27726799, -2.02071153],
                            [-1.12381617, -3.83857802],
                            [-2.28118934, -1.46633827],
                            [-2.3549158 , -1.80657578],
                            [-1.74044656, -2.07526724],
                            [-2.79704586, -1.36443595],
                            [-1.92218385, -2.93250477],
                            [-2.92701044, -2.96847166],
                            [-1.4253747 , -0.22921194],
                            [-0.74387201, -3.30511902],
                            [-1.15498944, -2.77366203],
                            [-2.24644488, -2.38313689],
                            [-1.90995148, -3.42895738],
                            [-2.51323051, -0.77033329],
                            [-2.35154454, -1.22348593],
                            [-3.17657854, -3.44646081],
                            [-2.09821233, -2.17331636],
                            [-2.46472047, -3.25665881],
                            [-2.93195696, -3.44441331],
                            [-1.21331688, -1.74102311],
                            [-4.26147205, -1.15408288],
                            [-3.92781729, -1.6466731 ],
                            [-2.61653796, -2.83036407],
                            [-2.13870779, -1.92183406],
                            [-0.92668419, -1.28751845],
                            [-1.48168474, -0.47637154],
                            [-1.32976769, -2.14975811],
                            [-2.38552262, -1.22810436],
                            [-1.28950925, -5.09365429],
                            [-2.14545802, -3.71160894],
                            [-0.77893222, -1.93809604],
                            [-3.79561367, -3.09868489],
                            [-3.78666878, -2.34335706],
                            [-2.97868314, -0.09441922],
                            [-2.1384787 , -1.38147996],
                            [-1.10693326, -1.4666833 ],
                            [-1.50229409, -3.53342199],
                            [-2.98972317, -1.90525036],
                            [-0.58440998, -2.20500573],
                            [-1.34894075, -2.64962571],
                            [-0.29279357, -1.50929611],
                            [-2.50315156, -2.30215839],
                            [-2.22635297, -1.37115595],
                            [-2.90236169, -1.85873759],
                            [-2.72270423, -2.80739472],
                            [-3.17455712, -0.412053  ],
                            [-2.41082405, -3.27370625],
                            [-3.61586485, -3.72268564]])

    // Sigmoid(wTx+b)
    def "should calculate probability for one point belonging to a class"() {
        given:
            Placeholder<Matrix> x = new Placeholder<>()
            Variable<Matrix> wT = new Variable<>(new Matrix((double[][])[
                [1.0, 1.0]
            ]))
            Variable<Matrix> b = new Variable<>(new Matrix((double[][])[
                [0.0]
            ]))
            Sigmoid p = new Sigmoid(new Add(new Multiply(wT, x), b))
        when:
            Matrix result = new Session<Matrix>().run(
                p,
                new HashMap<Placeholder<Matrix>, Matrix>() {{
                    put(x, new Matrix((double[][])[
                        [3.0],
                        [2.0]]
                    ))
                }}
            )
        then:
            assert result.rows == 1
            assert result.columns == 1
            assert result.data[0][0] > 0.99
    }

    // Softmax(XW+b)
    def "should calculate probabilities for multi class perceptron - bluePoints"() {
        given:
            Placeholder<Matrix> X = new Placeholder<>()
            Variable<Matrix> W = new Variable<>(new Matrix((double[][])[
                [1.0, -1.0],
                [1.0, -1.0]
            ]))
            Variable<Matrix> b = new Variable<>(new Matrix((double[][])[
                [0.0, 0.0]
            ]))
            Softmax p = new Softmax(new AddRowVectorForEachRow(new Multiply(X, W), b))
        when:
            Matrix result = new Session<Matrix>().run(
                p,
                new HashMap<Placeholder<Matrix>, Matrix>() {{
                    put(X, bluePoints)
                }}
            )
        then:
            // println result
            for(double[] row: result.data) {
                assert row[0] >= 0.8
                assert row[0] <= 1.0
                assert row[1] >= 0.0
                assert row[1] <= 0.2
            }
    }

    // Softmax(XW+b)
    def "should calculate probabilities for multi class perceptron - redPoints"() {
        given:
            Placeholder<Matrix> X = new Placeholder<>()
            Variable<Matrix> W = new Variable<>(new Matrix((double[][])[
                [1.0, -1.0],
                [1.0, -1.0]
            ]))
            Variable<Matrix> b = new Variable<>(new Matrix((double[][])[
                [0.0, 0.0]
            ]))
            Softmax p = new Softmax(new AddRowVectorForEachRow(new Multiply(X, W), b))
        when:
            Matrix result = new Session<Matrix>().run(
                p,
                new HashMap<Placeholder<Matrix>, Matrix>() {{
                    put(X, redPoints)
                }}
            )
        then:
            // println result
            for(double[] row: result.data) {
                assert row[0] >= 0.0
                assert row[0] <= 0.2
                assert row[1] >= 0.8
                assert row[1] <= 1.0
            }
    }

    def "should calculate negative log likelihood loss"() {
        given:
            Placeholder<Matrix> X = new Placeholder<>()
            Placeholder<Matrix> c = new Placeholder<>()
            Variable<Matrix> W = new Variable<>(new Matrix((double[][])[
                [1.0, -1.0],
                [1.0, -1.0]

            ]))
            Variable<Matrix> b = new Variable<>(new Matrix((double[][])[
                [0.0, 0.0]
            ]))
            Softmax p = new Softmax(new AddRowVectorForEachRow(new Multiply(X, W), b))
            Negative J = new Negative(new ReduceSum(new ReduceSum(new MultiplyElementwise(c, new Log(p)), 1)))
        when:
            Matrix result = new Session<Matrix>().run(
                J,
                new HashMap<Placeholder<Matrix>, Matrix>() {{
                    put(X, buildX())
                    put(c, buildC())
                }}
            )
        then:
            // println result
            assert result.getData()[0][0] > 0.0
    }

    private static Matrix buildX() {
        double[][] data = new double[100][2]
        for(int i = 0; i < bluePoints.data.length; i++) {
            data[i] = bluePoints.data[i]
        }
        for(int i = 0; i < redPoints.data.length; i++) {
            data[i + 50] = redPoints.data[i]
        }
        return new Matrix(data)
    }

    private static Matrix buildC() {
        double[][] data = new double[100][2]
        for(int i = 0; i < bluePoints.data.length; i++) {
            data[i] = (double[])[1.0d, 0.0d]
        }
        for(int i = 0; i < redPoints.data.length; i++) {
            data[i + 50] = (double[])[0.0d, 1.0d]
        }
        return new Matrix(data)
    }

}
