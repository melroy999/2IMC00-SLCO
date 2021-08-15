/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.UnaryBuiltInValueFunctionCall;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Unary Built In Value Function Call</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class UnaryBuiltInValueFunctionCallTest extends BuiltInValueFunctionCallTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(UnaryBuiltInValueFunctionCallTest.class);
	}

	/**
	 * Constructs a new Unary Built In Value Function Call test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public UnaryBuiltInValueFunctionCallTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Unary Built In Value Function Call test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected UnaryBuiltInValueFunctionCall getFixture() {
		return (UnaryBuiltInValueFunctionCall)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createUnaryBuiltInValueFunctionCall());
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

} //UnaryBuiltInValueFunctionCallTest
