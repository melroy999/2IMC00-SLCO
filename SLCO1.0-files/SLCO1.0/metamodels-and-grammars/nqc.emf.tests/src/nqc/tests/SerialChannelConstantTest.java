/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.SerialChannelConstant;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Serial Channel Constant</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class SerialChannelConstantTest extends ConstantExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(SerialChannelConstantTest.class);
	}

	/**
	 * Constructs a new Serial Channel Constant test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialChannelConstantTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Serial Channel Constant test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected SerialChannelConstant getFixture() {
		return (SerialChannelConstant)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createSerialChannelConstant());
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

} //SerialChannelConstantTest
