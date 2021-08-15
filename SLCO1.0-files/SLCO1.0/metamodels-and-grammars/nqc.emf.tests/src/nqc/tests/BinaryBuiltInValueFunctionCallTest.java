/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.BinaryBuiltInValueFunctionCall;
import nqc.NqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Binary Built In Value Function Call</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class BinaryBuiltInValueFunctionCallTest extends BuiltInValueFunctionCallTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(BinaryBuiltInValueFunctionCallTest.class);
	}

	/**
	 * Constructs a new Binary Built In Value Function Call test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BinaryBuiltInValueFunctionCallTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Binary Built In Value Function Call test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected BinaryBuiltInValueFunctionCall getFixture() {
		return (BinaryBuiltInValueFunctionCall)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createBinaryBuiltInValueFunctionCall());
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

} //BinaryBuiltInValueFunctionCallTest
