/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.SerialPacketConstant;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Serial Packet Constant</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class SerialPacketConstantTest extends ConstantExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(SerialPacketConstantTest.class);
	}

	/**
	 * Constructs a new Serial Packet Constant test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialPacketConstantTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Serial Packet Constant test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected SerialPacketConstant getFixture() {
		return (SerialPacketConstant)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createSerialPacketConstant());
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

} //SerialPacketConstantTest
