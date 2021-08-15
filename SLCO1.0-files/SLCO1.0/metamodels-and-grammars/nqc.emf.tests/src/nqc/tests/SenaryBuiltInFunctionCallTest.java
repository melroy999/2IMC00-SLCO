/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.SenaryBuiltInFunctionCall;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Senary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class SenaryBuiltInFunctionCallTest extends BuiltInFunctionCallTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(SenaryBuiltInFunctionCallTest.class);
	}

	/**
	 * Constructs a new Senary Built In Function Call test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SenaryBuiltInFunctionCallTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Senary Built In Function Call test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected SenaryBuiltInFunctionCall getFixture() {
		return (SenaryBuiltInFunctionCall)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createSenaryBuiltInFunctionCall());
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

} //SenaryBuiltInFunctionCallTest
