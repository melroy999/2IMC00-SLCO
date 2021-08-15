/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.OutputModeConstant;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Output Mode Constant</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class OutputModeConstantTest extends ConstantExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(OutputModeConstantTest.class);
	}

	/**
	 * Constructs a new Output Mode Constant test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutputModeConstantTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Output Mode Constant test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected OutputModeConstant getFixture() {
		return (OutputModeConstant)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createOutputModeConstant());
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

} //OutputModeConstantTest
