/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.BinaryBuiltInFunctionCall;
import nqc.NqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Binary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class BinaryBuiltInFunctionCallTest extends BuiltInFunctionCallTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(BinaryBuiltInFunctionCallTest.class);
	}

	/**
	 * Constructs a new Binary Built In Function Call test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BinaryBuiltInFunctionCallTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Binary Built In Function Call test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected BinaryBuiltInFunctionCall getFixture() {
		return (BinaryBuiltInFunctionCall)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createBinaryBuiltInFunctionCall());
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

} //BinaryBuiltInFunctionCallTest
