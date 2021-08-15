/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.SerialChecksumConstant;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Serial Checksum Constant</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class SerialChecksumConstantTest extends ConstantExpressionTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(SerialChecksumConstantTest.class);
	}

	/**
	 * Constructs a new Serial Checksum Constant test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SerialChecksumConstantTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Serial Checksum Constant test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected SerialChecksumConstant getFixture() {
		return (SerialChecksumConstant)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createSerialChecksumConstant());
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

} //SerialChecksumConstantTest
