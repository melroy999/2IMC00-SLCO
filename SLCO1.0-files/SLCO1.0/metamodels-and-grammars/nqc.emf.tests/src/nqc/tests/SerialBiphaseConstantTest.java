/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.SerialBiphaseConstant;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Serial Biphase Constant</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class SerialBiphaseConstantTest extends ConstantExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(SerialBiphaseConstantTest.class);
	}

	/**
	 * Constructs a new Serial Biphase Constant test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialBiphaseConstantTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Serial Biphase Constant test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected SerialBiphaseConstant getFixture() {
		return (SerialBiphaseConstant)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createSerialBiphaseConstant());
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

} //SerialBiphaseConstantTest
