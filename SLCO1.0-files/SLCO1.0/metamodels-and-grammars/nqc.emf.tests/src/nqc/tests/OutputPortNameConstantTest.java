/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.OutputPortNameConstant;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Output Port Name Constant</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class OutputPortNameConstantTest extends ConstantExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(OutputPortNameConstantTest.class);
	}

	/**
	 * Constructs a new Output Port Name Constant test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutputPortNameConstantTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Output Port Name Constant test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected OutputPortNameConstant getFixture() {
		return (OutputPortNameConstant)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createOutputPortNameConstant());
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#tearDown()
	 * @generated
	 */
	@Override
	protected void tearDown() throws Exception {
		setFixture(null);
	}

} //OutputPortNameConstantTest
